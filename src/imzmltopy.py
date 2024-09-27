# Transform .imzml file to matrix
# Melanie NIJS

import numpy as np
import xml.etree.ElementTree as ET
from collections import namedtuple
import time
import os
from binning import binning
from utils import execute_indexed_parallel_mp
import multiprocessing as mp


def correction(signal):
    median = np.nanmedian(signal)
    signal -= median
    signal[signal < 0] = 0
    return signal
 
# iterating over directory and subdirectory to get desired result
def convert(path, path_save, dtype, low_mz, high_mz, limit):
    # giving file extensions
    ext = ('.imzML')
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # iterating over directory and subdirectory to get desired result
        for _, _, f in os.walk(path):
            for name in f:
                if name.endswith(ext):
                    fname = os.path.join(path, os.path.splitext(name)[0])
                    start = time.time()
                    print('.imzML and .ibd file = ', fname)
                    
                    # read .imzl file and extract pixels and spectra
                    hupostr = '{http://psi.hupo.org/ms/mzml}'
                    spectrumtup = namedtuple('spectrumtup', ['x', 'y', 'mzs', 'intensities'])
                    datatup = namedtuple('datatup', ['length', 'encoded', 'offset'])

                    def spectrum2dict(e):
                        x = None
                        y = None

                        scanlist = e.find(hupostr + 'scanList')
                        scan = scanlist.find(hupostr + 'scan')
                        for cvpar in scan.iter(hupostr + 'cvParam'):
                            if cvpar.attrib['name'] == 'position x':
                                x = int(cvpar.attrib['value'])
                            if cvpar.attrib['name'] == 'position y':
                                y = int(cvpar.attrib['value'])

                        bdlst = e.find(hupostr + 'binaryDataArrayList').findall(hupostr + 'binaryDataArray')
                        mzselem, spectelem = bdlst  # fixme: not robust enough, mzarray not necessarily first

                        for cvpar in mzselem.iter(hupostr + 'cvParam'):
                            if cvpar.attrib['name'] == 'external array length':
                                mzlength = int(cvpar.attrib['value'])
                            if cvpar.attrib['name'] == 'external encoded length':
                                mzencoded = int(cvpar.attrib['value'])
                            if cvpar.attrib['name'] == 'external offset':
                                mzoffset = int(cvpar.attrib['value'])
                        mzs = datatup(length=mzlength, encoded=mzencoded, offset=mzoffset)

                        for cvpar in spectelem.iter(hupostr + 'cvParam'):
                            if cvpar.attrib['name'] == 'external array length':
                                intlength = int(cvpar.attrib['value'])
                            if cvpar.attrib['name'] == 'external encoded length':
                                intencoded = int(cvpar.attrib['value'])
                            if cvpar.attrib['name'] == 'external offset':
                                intoffset = int(cvpar.attrib['value'])
                        intensities = datatup(length=intlength, encoded=intencoded, offset=intoffset)

                        return spectrumtup(x=x, y=y, mzs=mzs, intensities=intensities)

                    def save_data():

                        xmltree = ET.parse(fname + '.imzML')
                        xmlroot = xmltree.getroot()
                        runkey = '{http://psi.hupo.org/ms/mzml}run'
                        run = xmlroot.find(runkey)
                        spectrumlistkey = '{http://psi.hupo.org/ms/mzml}spectrumList'
                        spectrumlist = run.find(spectrumlistkey)
                        spectrumkey = '{http://psi.hupo.org/ms/mzml}spectrum'
                        spectraelems = spectrumlist.findall(spectrumkey)
                    
                        imzml_data = list(map(spectrum2dict, spectraelems))
                        
                        mzs = list(map(lambda x: np.memmap(filename=fname + '.ibd', dtype='float32', mode='c', offset=x.mzs.offset, shape=(x.mzs.length,)), imzml_data))
                        shape = int(np.round(np.mean(np.array(list(map(lambda x: np.argmin(np.abs(high_mz - x) - np.argmin(np.abs(low_mz - x))), mzs))))))
                        print(f'Measured spectra are between {np.min(np.array(list(map(lambda x: np.min(x), mzs))))} Da and {np.max(np.array(list(map(lambda x: np.max(x), mzs))))} Da.')
                        mz_vector = np.arange(shape) / (shape - 1) * (high_mz - low_mz) + low_mz
                        
                        intensities = list(map(lambda x: np.memmap(filename=fname + '.ibd', dtype='float32', mode='c', offset=x.intensities.offset, shape=(x.intensities.length,)), imzml_data))
                        
                        data_matrix = np.ndarray((len(mzs), shape), dtype=intensities[0].dtype)
                        
                        # Multiprocessing
                        result = execute_indexed_parallel_mp(pool, binning, args=[(mz_vector, mzs[i], intensities[i], limit) for i in range(len(mzs))])
                        
                        for i, d in result:
                            data_matrix[i, :] = d
                        
                        row2grid = np.array(list(map(lambda p: (p.x, p.y), imzml_data)), dtype=int)

                        if dtype != data_matrix.dtype:
                            data_matrix = dtype(data_matrix)
                            
                        for i in range(data_matrix.shape[0]):
                            data_matrix[i, :] = correction(data_matrix[i, :])
                            if np.isclose(np.sum(data_matrix[i, :]), 0, atol=1e-4):
                                data_matrix[i, :] = np.ones_like(data_matrix[i, :], dtype=data_matrix.dtype)

                        np.save(f"{path_save}/{os.path.splitext(name)[0]}_noTIC_matrix.npy", data_matrix)
                        print('Shape matrix =', (len(imzml_data), data_matrix.shape[1]))

                        np.save(f"{path_save}/{os.path.splitext(name)[0]}_row2grid.npy", row2grid)
                        np.save(f"{path_save}/{os.path.splitext(name)[0]}_mz_vector.npy", mz_vector if dtype == mz_vector.dtype else dtype(mz_vector))
                                
                    save_data()
                            
                    # Uncomment to save memory (removes the imzML and ibd files after convertion)
                    # os.remove(f'{path}/{name.replace(".imzML", ".ibd")}')
                    # os.remove(f'{path}/{name}')

                    end = time.time()
                    timeVal = end - start
                    print('finished in ' + str(timeVal) + ' seconds.')
                    
                    
if __name__ == '__main__':
    
    path = f'{os.getcwd()}/data/raw'
    path_save = f'{os.getcwd()}/data/numpy'
    dtype = np.float32
    limit = 3000
    low_mz = 100
    high_mz = 1000
    
    convert(path, path_save, dtype, low_mz, high_mz, limit)
             
