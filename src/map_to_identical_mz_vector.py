import multiprocessing as mp
import numpy as np
import os

import fnmatch
from binning import binning_matrix
from copy import deepcopy
from tqdm import tqdm
from utils import execute_indexed_parallel_mp


def get_best_mz_vector(path: str='./data/python/float32/original'):
    
    min_mz = 0
    max_mz = np.inf
    files = fnmatch.filter(sorted(os.listdir(path)), '*_mz_vector.npy')
    for file in files:
        mz_vector = np.load(f'{path}/{file}')
        if mz_vector[0] > min_mz:
            min_mz = mz_vector[0]
        if mz_vector[-1] < max_mz:
            max_mz = mz_vector[-1]
    
    counts = []
    for file in files:
        mz_vector = np.load(f'{path}/{file}')
        index_min = np.argmin(np.abs(mz_vector - min_mz))
        index_max = np.argmin(np.abs(mz_vector - max_mz))
        counts.append(index_max - index_min)

    index_sample = np.argmin(counts)
    
    mz_vector = np.load(f'{path}/{files[index_sample]}')
    index_min_mz = np.argmin(np.abs(mz_vector - min_mz))
    index_max_mz = np.argmin(np.abs(mz_vector - max_mz))
    
    return mz_vector[index_min_mz: index_max_mz + 1]
    

def map_to_identical_mz_vector(data: np.ndarray, mz_vector: np.ndarray, best_mz_vector: np.ndarray):
    return binning_matrix(best_mz_vector, mz_vector, data, limit=3000)
    
def get_samples(path: str):
    
    files = list(filter(lambda x: os.path.isfile(f'{path}/{x}') and x.endswith('_row2grid.npy'), os.listdir(path)))
    return [file.split('_row2grid.npy')[0] for file in files]


if __name__ == '__main__':
    
    path = f'{os.getcwd()}/data/numpy/'
    
    save_path = f'{path}/aligned/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    best_mz_vector = get_best_mz_vector(path)
    
    np.save(f'{save_path}/mz_vector.npy', best_mz_vector)
    
    for sample in get_samples(path):
        print(f'Sample: {sample}')
        data = np.load(f'{path}/{sample}_noTIC_matrix.npy')
        mz_vectors = np.load(f'{path}/{sample}_mz_vector.npy')
        row2grid = np.load(f'{path}/{sample}_row2grid.npy')
        
        data = map_to_identical_mz_vector(data, mz_vectors, best_mz_vector)
        
        np.save(f'{save_path}/{sample}_noTIC_matrix.npy', data)
        np.save(f'{save_path}/{sample}_row2grid.npy', row2grid)
