import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import compute_all_metrics, masked_mape, masked_rmse
from tqdm import tqdm

class szy_Engine(BaseEngine):
    def __init__(self, **args):
        super(szy_Engine, self).__init__(**args)
        self.separate_loss = nn.TripletMarginLoss(margin=1.0)
        self.compact_loss = nn.MSELoss()

    def ent_and_link_loss(self, supports, ass_list):
        all_link_loss = 0
        all_ent_loss = 0
        if self._args.spatial_scale > 1:
            for i in range(1, self._args.spatial_scale):
                ass = F.softmax(ass_list[i-1], dim=-1)
                adj = supports[i-1][0]
                link_loss = adj - torch.matmul(ass, ass.transpose(0, 1))
                link_loss = torch.norm(link_loss, p=2) / self._args.node_num
                ent_loss = (-ass * torch.log(ass + 1e-15)).sum(dim=-1).mean()
                all_link_loss += link_loss
                all_ent_loss += ent_loss
        return all_ent_loss + all_link_loss

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in tqdm(self._dataloader['train_loader'].get_iterator()):
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred, h_att, query, pos, neg, supports, ass_list = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            loss1 = self._loss_fn(pred, label, mask_value)
            loss2 = self.separate_loss(query, pos.detach(), neg.detach())
            loss3 = self.compact_loss(query, pos.detach())
            loss4 = self.ent_and_link_loss(supports, ass_list)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss = loss1 + \
                self._args.lamb * loss2 + \
                    self._args.lamb1 * loss3 + \
                        self._args.lamb2 * loss4
            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss1.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)
    
    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred, _, _, _, _, _, _ = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))



            