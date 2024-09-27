import h5py
import os
import numpy as np

def read_h5(path):
    with h5py.File(path, "r") as f:
        intensities = f['Data'][()]
        mz_vector = f['mzArray'][()]
        locations = np.stack([f['xLocation'][()], f['yLocation'][()]], axis=1)
    return intensities, mz_vector, locations
        
if __name__ == '__main__':
    
    path = "./data/ST002045_massNet/massNet_Raw_h5/"

    for filename in os.listdir(path):
        read_h5(f'{path}/{filename}')
        
    