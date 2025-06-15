import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMS04':
        data_path = os.path.join('/home/wjj/projects/STG/data/PEMS04/PEMS04.npz')
        # data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, :]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS08':
        data_path = os.path.join('/home/wjj/projects/STG/data/PEMS08/data.npz')
        data = np.load(data_path)['data']  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS03':
        data_path = os.path.join('../data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS07':
        data_path = os.path.join('/home/wjj/projects/STG/data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7M':
        data_path = os.path.join('../data/PEMS07M/PEMS07M.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7L':
        data_path = os.path.join('../data/PEMS07L/PEMS07L.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'Decentraland':
        data_path = os.path.join('../token_data/Decentraland_node_features.npz')
        data = np.load(data_path)['arr_0'][:, :, 0]  #1 dimension, degree
    elif dataset == 'Bytom':
        data_path = os.path.join('../token_data/Bytom_node_features.npz')
        data = np.load(data_path)['arr_0'][:, :, 0]  #1 dimension, degree
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
