import torch
import os
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics
import shap
from src.models.mjm_shap import STRIP_shap

class XYQ_Engine(BaseEngine):
    def __init__(self, cl_step, warm_step, **args):
        super(XYQ_Engine, self).__init__(**args)
        self._cl_step = cl_step
        self._warm_step = warm_step
        self._horizon = self._args.horizon
        self._cl_len = 0

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_c{}_same{}.pt'.format(self._seed, 
                                                          self._args.core, 
                                                          self._args.same)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_s{}_c{}_same{}.pt'.format(self._seed, 
                                                          self._args.core, 
                                                          self._args.same)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename), map_location=self._device))  

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        for X, label in tqdm(self._dataloader['train_loader'].get_iterator()):
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('check mask value', mask_value)

           
            # gradually training step
            self._iter_cnt += 1
            '''
            if self._iter_cnt < self._warm_step:
                self._cl_len = self._horizon
            elif self._iter_cnt == self._warm_step:
                self._cl_len = 1
            else:
                if (self._iter_cnt - self._warm_step) % self._cl_step == 0 and self._cl_len < self._horizon:
                    self._cl_len += 1

            pred = pred[:, :self._cl_len, :, :]
            label = label[:, :self._cl_len, :, :]
            '''

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), None
    
    def evaluate_shap(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        args = self._args
        model = STRIP_shap(node_num=args.node_num,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 model_args=vars(args),
                 stmodel=self.model.stmodel,
                 dim=[128, 128],
                 core=args.core,
                 ).to(self._device)
        filename = 'final_model_s{}_c{}_same{}.pt'.format(self._seed, 
                                                          self._args.core, 
                                                          self._args.same)
        model.load_state_dict(torch.load(
            os.path.join(self._save_path, filename), map_location=self._device))  
        




        preds = []
        labels = []

        flucts = []
        stabs = []
        corrs = []
        errors = []
        days = []
        weeks = []
        nodes = []
        index = 0
        with torch.no_grad():
            for X, label in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                fluct, stab, corr, pred, day, week, node = self.model(X, label)
                error = label - pred
                # pred, label = self._inverse_transform([pred, label])
                flucts.append(fluct)
                stabs.append(stab)
                corrs.append(corr)
                errors.append(error)
                days.append(day)
                weeks.append(week)
                nodes.append(node)
                preds.append(pred)
                labels.append(label)
                '''
                flucts.append(fluct.cpu())
                stabs.append(stab.cpu())
                corrs.append(corr.cpu())
                errors.append(error.cpu())
                days.append(day.cpu())
                weeks.append(week.cpu())
                nodes.append(node.cpu())
                preds.append(pred.cpu())
                labels.append(label.cpu())
                '''
                index += 1
                if index == 2:
                    break
        
        
        flucts = torch.cat(flucts, dim=0)
        stabs = torch.cat(stabs, dim=0)
        corrs = torch.cat(corrs, dim=0)
        errors = torch.cat(errors, dim=0)
        days = torch.cat(days, dim=0)
        weeks = torch.cat(weeks, dim=0)
        nodes = torch.cat(nodes, dim=0)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        
        explainer = shap.DeepExplainer(model, 
                                         [flucts[:64], 
                                          stabs[:64], 
                                          corrs[:64], 
                                          errors[:64],
                                          days[:64],
                                          weeks[:64],
                                          nodes[:64]])
        print(explainer)
        ob = -1
        shap_values = explainer.shap_values([flucts[64:ob], 
                                             stabs[64:ob], 
                                             corrs[64:ob], 
                                             errors[64:ob],
                                             days[64:ob],
                                             weeks[64:ob],
                                             nodes[64:ob]])
        for i in range (len(shap_values)):
            np.save('shap_' + str(i) + '.npy', shap_values[i])
        '''
        shap.force_plot(explainer.expected_value, 
                        shap_values, 
                        [flucts[64:], 
                         stabs[64:], 
                         corrs[64:], 
                         errors[64:]]
                        )
        '''