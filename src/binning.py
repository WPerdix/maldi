import numpy as np
from bisect import bisect


def binning(x_correct, x_wrong, values, limit=10):
    
    index_min = np.argmin(np.abs(x_wrong - x_correct[0]))
    if x_wrong[index_min] > x_correct[0]:
        index_min -= 1
        index_min = np.max([index_min, 0])
    index_max = np.argmin(np.abs(x_wrong - x_correct[-1]))
    if x_wrong[index_min] < x_correct[-1]:
        index_max += 1
        index_max = np.min([index_max, x_wrong.shape[0]])
        
    result = np.zeros_like(x_correct, dtype=values.dtype)
    
    values = values[index_min: index_max + 1]
        
    index = None
    max_index = x_correct.shape[0] - 1
    for i, x in enumerate(x_wrong[index_min: index_max + 1]):
        if index:
            index = bisect(x_correct, x, index, np.min([index + limit, max_index + 1])) - 1
        else:
            index = bisect(x_correct, x) - 1
        if index < 0:
            result[0] += values[i] * (x - x_correct[0]) / (x_correct[0] - x_correct[1])
            index = 0
        elif index >= x_correct.shape[0] - 1:
            result[-1] += values[i] * (x - x_correct[index]) / (x_correct[index] - x_correct[index - 1])
            index = x_correct.shape[0] - 1
        else:                
            factor = (x - x_correct[index]) / (x_correct[index + 1] - x_correct[index])
            result[index] += values[i] * factor
            result[index + 1] += values[i] * (1 - factor)
            if factor > 1:
                print('Limit should be higher!')
    return result

def binning_matrix(x_correct, x_wrong, values, axis=1, limit=10):
    
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
        
    index = None
    max_index = x_correct.shape[0] - 1
    for i, x in enumerate(x_wrong[index_min: index_max + 1]):
        if index:
            index = bisect(x_correct, x, index, np.min([index + limit, max_index + 1])) - 1
        else:
            index = bisect(x_correct, x) - 1
        if index < 0:
            result[:, 0] += values[:, i] * (x - x_correct[0]) / (x_correct[0] - x_correct[1])
            index = 0
        elif index >= x_correct.shape[0] - 1:
            result[:, -1] += values[:, i] * (x - x_correct[index]) / (x_correct[index] - x_correct[index - 1])
            index = x_correct.shape[0] - 1
        else:
            factor = (x - x_correct[index]) / (x_correct[index + 1] - x_correct[index])
            result[:, index] += values[:, i] * factor
            result[:, index + 1] += values[:, i] * (1 - factor)
            if factor > 1:
                print('Limit should be higher!')
    return result


if __name__ == "__main__":
    a = np.arange(10) * 2
    b = np.arange(5) * 2 + 1
    values_a = 2*np.copy(a) + 2
    values_b = 2*np.copy(b) + 2
    
    assert (values_b == binning(b, a, values_a)).all()
    
    a = np.arange(10) * 2
    b = np.arange(5) * 2 + 1
    values_a = 2*np.copy(a) + 2
    values_b = 2*np.copy(b) + 2
    result = values_a
    result[b.shape[0] + 1:] = np.zeros_like(result[b.shape[0] + 1:])
    result[b.shape[0]] = 10

    assert (result == binning(a, b, values_b)).all()
    
    a = np.arange(10) * 2
    b = np.arange(5) * 2 + 5
    values_a = 2*np.copy(a) + 2
    values_b = 2*np.copy(b) + 2

    assert (values_b == binning(b, a, values_a)).all()
    
    a = np.arange(10) * 2
    b = np.arange(5) * 2 + 5
    values_a = 2*np.copy(a) + 2
    values_b = 2*np.copy(b) + 2
    
    temp = 4 * np.arange(5) + 12
    result = np.zeros_like(values_a)
    ind = 2
    for i, x in enumerate(temp):
        result[ind + i] += x / 2
        result[ind + i + 1] += x / 2

    assert (result == binning(a, b, values_b)).all()    
