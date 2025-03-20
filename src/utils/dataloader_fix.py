import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, input_dim, pad_last_sample=True):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        # print('Defalut data shape:', data.shape)
        self.data = data
        # print('Practice data shape:', self.data.shape)
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon
        self.input_dim = input_dim

    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :self.input_dim]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], min(self.data.shape[-1], self.input_dim))
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) if array_size == 1 else len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(data_path, args, logger):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'), 
                  allow_pickle=True)
    logger.info('Data shape: ' + str(ptr['data'].shape))
    
    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx, \
                                                 args.seq_len, args.horizon, args.bs if cat == 'train' else 1, 
                                                 logger, args.input_dim)

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    adj = np.load(numpy_file)
    np.fill_diagonal(adj, 1.)
    return adj


def get_dataset_info(dataset):
    base_dir1 = os.getcwd() + '/data/'
    base_dir2 = os.getcwd() + '/data_knowair/'
    base_dir3 = os.getcwd() + '/datagagnn/'
    base_dir4 = os.getcwd() + '/data_PeMS0X/'
    base_dir5 = '/home/mjm/LST/LargeST-main/AAAI2025/LargeST/data/'
    base_dir6 = os.getcwd() + '/data_METR-LA/'
    base_dir7 = os.getcwd() + '/data_SZEV/station/'
    base_dir8 = '/home/mjm/LST/LargeST-main/AAAI2025/Xtraffic/data/'
    base_dir9 = '/home/mjm/LST/AAA/data_Milan_Trentino/'

    d = {
         '24_24': [base_dir1 + dataset, base_dir1+'adj_mx.npy', 1341],
         '24_24_KA': [base_dir2 + '24_24', base_dir2+'adj_mx.npy', 184],
         '24_24_G': [base_dir3 + '24_24', base_dir3+'adj_mx.npy', 209],
         '3': [base_dir4 + dataset, base_dir4+'PEMS03_adj.npy', 358, 3, 288, 12, 12],
         '4': [base_dir4 + dataset, base_dir4+'PEMS04_adj.npy', 307, 5, 288, 12, 12],
         '7': [base_dir4 + dataset, base_dir4+'PEMS07_adj.npy', 883, 3, 288, 12, 12],
         '8': [base_dir4 + dataset, base_dir4+'PEMS08_adj.npy', 170, 5, 288, 12, 12],
         'la': [base_dir6 + dataset,  base_dir6 + 'la/la_adj.npy', 207, 3, 288, 12, 12],
         'bay': [base_dir6 + dataset, base_dir6 + 'bay/bay_adj.npy', 325, 3, 288, 12, 12],
         'sd': [base_dir5 + dataset, base_dir5 + dataset + '/sd_rn_adj.npy', 716, 3, 96, 12, 12],
         'gba': [base_dir5 + dataset, base_dir5 + dataset + '/gba_rn_adj.npy', 2352, 3, 96, 12, 12],
         'gla': [base_dir5 + dataset, base_dir5 + dataset + '/gla_rn_adj.npy', 3834, 3, 96, 12, 12],
         'ca': [base_dir5 + dataset, base_dir5 + dataset + '/ca_rn_adj.npy', 8600, 3, 96, 12, 12],
         'szev': [base_dir7, base_dir7 + 'adj_mx.npy', 1682, 5, 24, 24, 24],
         'xtraffic': [base_dir8, base_dir8 + '2023/adj_diag_1.npy', 16972, 3, 96, 12, 12],
         'call': [base_dir9 + dataset, base_dir9 + 'adj_mx.npy', 10000, 3, 24, 24, 24],
         'net': [base_dir9 + dataset, base_dir9 + 'adj_mx.npy', 10000, 3, 24, 24, 24],
         'sms': [base_dir9 + dataset, base_dir9 + 'adj_mx.npy', 10000, 3, 24, 24, 24,],
        }
    assert dataset in d.keys()
    return d[dataset]

def metapath(dataset):
    base_dir1 = os.getcwd() + '/data/'
    base_dir2 = os.getcwd() + '/data_knowair/'
    base_dir3 = os.getcwd() + '/datagagnn/'
    base_dir4 = os.getcwd() + '/data_PeMS0X/'
    base_dir5 = '/home/mjm/LST/LargeST-main/AAAI2025/LargeST/data/'
    base_dir7 = os.getcwd() + '/data_SZEV/station/'
    
    base_dir9 = '/home/mjm/LST/AAA/data_Milan_Trentino/'

    d = {
         '24_24': [base_dir1 + dataset, base_dir1+'adj_mx.npy', 1341],
         '24_24_KA': [base_dir2 + '24_24', base_dir2+'adj_mx.npy', 184],
         '24_24_G': [base_dir3 + '24_24', base_dir3+'adj_mx.npy', 209],
         '3': [base_dir4 + dataset, base_dir4+'PEMS03_adj.npy', 358, 3, 288],
         '4': [base_dir4 + dataset, base_dir4+'PEMS04_adj.npy', 307, 5, 288],
         '7': [base_dir4 + dataset, base_dir4+'PEMS07_adj.npy', 883, 3, 288],
         '8': [base_dir4 + dataset, base_dir4+'PEMS08_adj.npy', 170, 5, 288],
         'sd': [base_dir5 + dataset + '/' + dataset + '_meta.csv'],
         'gba': [base_dir5 + dataset + '/' + dataset + '_meta.csv'],
         'gla': [base_dir5 + dataset + '/' + dataset + '_meta.csv'],
         'ca': [base_dir5 + dataset + '/' + dataset + '_meta.csv'],
         'szev': [base_dir7 + '/meta.csv'],
         'call': [base_dir9 + 'meta.csv'],
         'net': [base_dir9 + 'meta.csv'],
         'sms': [base_dir9 + 'meta.csv'],
        }
    assert dataset in d.keys()
    return d[dataset]

def load_meta(dataset):
    base_dir1 = os.getcwd() + '/data/'
    base_dir2 = os.getcwd() + '/data_knowair/'
    base_dir3 = os.getcwd() + '/datagagnn/'
    base_dir4 = os.getcwd() + '/data_PeMS0X/'
    base_dir5 = '/home/mjm/LST/LargeST-main/AAAI2025/LargeST/data/'
    base_dir6 = os.getcwd() + '/data_METR-LA/'
    base_dir7 = os.getcwd() + '/data_SZEV/station/'
    base_dir8 = '/home/mjm/LST/LargeST-main/AAAI2025/Xtraffic/data/'
    base_dir9 = '/home/mjm/LST/AAA/data_Milan_Trentino/'

    d = {
        'sd': '/home/mjm/LST/AAA/data_largest/sd/process_sd_meta.npy',
        }
    assert dataset in d.keys()
    return np.load(d[dataset])