import h5py
import os
import numpy as np

def read_h5(path):
    with h5py.File(path, "r") as f:
        intensities = f['Data'][()]
        mz_vector = f['mzArray'][()]
        locations = np.int32(np.stack([f['xLocation'][()], f['yLocation'][()]], axis=1))
    return intensities.T, mz_vector, locations
        
if __name__ == '__main__':
    
    path = "./data/ST002045_massNet/massNet_Raw_h5/"

    for filename in os.listdir(path):
        intensities, mz_vector, row2grid = read_h5(f'{path}/{filename}')
        print(filename)
        print('\tShape data matrix:', intensities.shape)
        print('\tShape mz_vector:', mz_vector.shape)
        print('\tShape location matrix:', row2grid.shape)
        
    