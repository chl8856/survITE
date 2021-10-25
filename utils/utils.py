import numpy as np
import random


def f_get_minibatch(mb_size, x1, x2=None, x3=None, x4=None, x5=None):
    idx = range(np.shape(x1)[0])
    idx = random.sample(idx, mb_size)
    
    if x2 is None:
        return x1[idx].astype(float)
    if x3 is None:
        return x1[idx].astype(float), x2[idx].astype(float)
    if x4 is None:
        return x1[idx].astype(float), x2[idx].astype(float), x3[idx].astype(float)
    if x5 is None:
        return x1[idx].astype(float), x2[idx].astype(float), x3[idx].astype(float), x4[idx].astype(float)
    
    return x1[idx].astype(float), x2[idx].astype(float), x3[idx].astype(float), x4[idx].astype(float), x5[idx].astype(float)