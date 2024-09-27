import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib as mpl
import numpy as np
import colorcet as cc
import os
import math
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from map_to_identical_mz_vector import get_samples

            
def make_image_color(row2grid, data):
    xmax = np.max(row2grid[:, 0])
    xmin = np.min(row2grid[:, 0])
    ymax = np.max(row2grid[:, 1])
    ymin = np.min(row2grid[:, 1])
    
    image_matrix = np.zeros([xmax - xmin + 1, ymax - ymin + 1, 3])
    for i, e in enumerate(row2grid):
        image_matrix[e[0] - xmin, e[1] - ymin, :] = data[i, :]
    return image_matrix

def scale_to_rgb(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    
    scaled_matrix = np.zeros_like(data)
    
    for column in range(data.shape[1]):
        scaled_matrix[:, column] = (data[:, column] - min_val[column]) / (max_val[column] - min_val[column])
    return scaled_matrix
    
def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    
def make_image(row2grid, data):
    xmax = np.max(row2grid[:, 0])
    xmin = np.min(row2grid[:, 0])
    ymax = np.max(row2grid[:, 1])
    ymin = np.min(row2grid[:, 1])

    image_matrix = np.zeros([xmax - xmin + 1, ymax - ymin + 1])
    for i, e in enumerate(row2grid):
        image_matrix[e[0] - xmin, e[1] - ymin] = data[i]
    return image_matrix
    
def make_ion_image(data, mz_vector, row2grid, index):
    if 0 <= index < mz_vector.shape[0]:
        mz_vector = np.ndarray.flatten(mz_vector)
        return make_image(row2grid, data[:, index]), mz_vector[index]

def make_image_color_clusters(row2grid, spatial_i, colors):
    xmax = np.max(row2grid[:, 0])
    xmin = np.min(row2grid[:, 0])
    ymax = np.max(row2grid[:, 1])
    ymin = np.min(row2grid[:, 1])

    # background is white 
    image_matrix = np.zeros([xmax - xmin + 1, ymax - ymin + 1, 3])
    for i, e in enumerate(row2grid):
        image_matrix[e[0] - xmin, e[1] - ymin] = colors[spatial_i[i]]

    return image_matrix
         
         
if __name__ == '__main__':
    path = os.getcwd() + "/data/numpy/"
    
    for sample in get_samples(path):
        row2grid = np.load(f"{path}/{sample}_row2grid.npy")
        mz_vector = np.load(f"{path}/{sample}_mz_vector.npy")
        data = np.load(f"{path}/{sample}_noTIC_matrix.npy")
        data[data <= 0] = 0
        
        for i in range(data.shape[0]):
            data[i, :] /= np.sum(data[i, :])
        
        # Uncomment to observe data between start Da and stop Da
        # start = 100
        # stop = 1000
        # from bisect import bisect_left, bisect_right
        # start_index = np.max([bisect_left(mz_vector, start) - 1, 0])
        # stop_index = np.min([bisect_right(mz_vector, stop), mz_vector.shape[0] - 1])
        # data = data[:, start_index: stop_index + 1]
        # mz_vector = mz_vector[start_index: stop_index + 1]

        pixel_to_index_dict = dict()
        xmin = np.min(row2grid[:, 0])
        ymin = np.min(row2grid[:, 1])

        for i, e in enumerate(row2grid):
            pixel_to_index_dict[e[0] - xmin, e[1] - ymin] = i
            
        global initialized
        initialized = False
        
        fig1, ax1 = plt.subplots(1, 1)
        fig2, (ax2, ax3) = plt.subplots(1, 2)
        
        def plot_sample(row2grid, data):
            ax2.imshow(make_image_color(row2grid, data), cmap=cc.cm.rainbow)
            ax2.set_title(f"Dimensionality reduction {sample}")
            
        def mouse_click(event):
            global initialized
            x, y = event.xdata, event.ydata
            if x and y:
                index = pixel_to_index_dict.get((np.round(y), np.round(x)))
                if index:
                    xlim, ylim = ax3.get_xlim(), ax3.get_ylim()
                    ax3.cla()
                    ax3.plot(mz_vector, data[index, :])
                    if initialized:
                        ax3.set_xlim(xlim)
                        ax3.set_ylim(ylim)
                    else:
                        initialized = True
                    ax3.set_xlabel('m/z value')
                    ax3.set_ylabel('intensity')
                    ax3.set_title('Mass Spectrum')
                    plt.pause(0.005)
        
        # reducer = PCA(n_components=3)
        reducer = UMAP(n_components=3)

        embedding = reducer.fit_transform(data)

        scaled_embedding = scale_to_rgb(embedding)

        plt.connect('button_press_event', mouse_click)
        plot_sample(row2grid, scaled_embedding)
        
        img, mz = make_ion_image(data, mz_vector, row2grid, int(mz_vector.shape[0] / 2))
        ax1.imshow(img)
        ax1.set_title(f'{"{:0.3f}".format(mz)} Da')
        axmz = fig1.add_axes([0.25, 0.1, 0.65, 0.03])
        # Make a horizontal slider to control the m/z values.
        mz_slider = Slider(
            ax=axmz,
            label='m/z index',
            valmin=0,
            valmax=mz_vector.shape[0] - 1,
            valinit=int(mz_vector.shape[0] / 2),
            valstep=1
        )
        
        def update(val):
            ax1.cla()
            img, mz = make_ion_image(data, mz_vector, row2grid, mz_slider.val)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            ax1.imshow(img)
            ax1.set_title(f'{"{:0.3f}".format(mz)} Da')
            fig1.canvas.draw_idle()
        
        fig1.subplots_adjust(bottom=0.25)
        mz_slider.on_changed(update)
        
        mz_slider.reset()
        
        reducer = KMeans(n_clusters=2)
        
        # Add black background colour to colormap
        colormap = mpl.colormaps['Accent']
        accent_colors = colormap.colors
        colors = [(0, 0, 0)]
        [colors.append(np.array(c)) for c in accent_colors]
        colormap.colors = np.array(colors)
        
        # + 1 so the background has a different colour
        labels = reducer.fit(data).labels_ + 1
        
        plt.figure()
        plt.imshow(make_image_color_clusters(row2grid, labels, colors)) 
        
        plt.show()
        