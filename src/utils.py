import numba as nb


def binning_matrix(x_correct, x_wrong, values):
    
    index_min = np.argmin(np.abs(x_wrong - x_correct[0]))
    if x_wrong[index_min] > x_correct[0]:
        index_min -= 1
        index_min = np.max([index_min, 0])
    index_max = np.argmin(np.abs(x_wrong - x_correct[-1]))
    if x_wrong[index_min] < x_correct[-1]:
        index_max += 1
        index_max = np.min([index_max, x_wrong.shape[0]])
        
    result = np.zeros((values.shape[0], x_correct.shape[0]), dtype=values.dtype)
    
    values = values[:, index_min: index_max + 1]
    x_wrong = x_wrong[index_min: index_max + 1]
    
    indices = np.searchsorted(x_correct, x_wrong, side='right') - 1
    
    idxs = indices < 0
    if idxs.sum() > 0:
        result[:, 0] = np.dot(values[:, idxs], x_wrong[idxs] - x_correct[0]) / (x_correct[0] - x_correct[1])
    
    idxs = np.logical_and(0 <= indices, indices < x_correct.shape[0] - 1)
    if idxs.sum() > 0:
        temp = indices[idxs]
        shifted = temp + 1
        factor = np.divide(x_wrong[idxs] - x_correct[temp], x_correct[shifted] - x_correct[temp])
        vals = values[:, idxs]
        for i, index in enumerate(temp):
            result[:, index] += vals[:, i] * (1 - factor[i])
            result[:, index + 1] += vals[:, i] * factor[i]
    
    idxs = indices >= x_correct.shape[0] - 1
    if idxs.sum() > 0:
        temp = indices[idxs]
        result[:, -1] += np.dot(values[:, idxs], np.divide(x_wrong[idxs] - x_correct[temp], x_correct[temp] - x_correct[temp - 1]))
        
    return result

