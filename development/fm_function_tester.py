'''
test program for calculating FM predicted value and gradient for three parameters:
    - bias b, linear weight vector w_i, and factor matrix V
'''


import numpy as np
from pyspark.ml.feature import CountVectorizer

def predictCTR(pair):
    """
        Compute the predicted probability AND return the gradient (?)
        Args:
            pair - records are in (label, sparse feature set) format
        Broadcast:
            b - bias term (scalar)
            w - linear weight vector (array)
            k - number of factors (def=2)
            V - factor matrix of size (d dimensions, k=2 factors)
        Returns:
            predRDD - pair of ([label, predicted probability], feature set)
    """
    
    feats = pair[1]
    
    # start with linear weight dot product
    linear_sum = 0.0
    for i in feats.indices:
        i = int(i)
        linear_sum += w_br.value[i]*feats[i]

    # factor matrix interaction sum
    factor_sum = 0.0
    lh_factor = 0.0
    rh_factor = 0.0
    
    for f in range(0, k_br.value):
        
        for i in feats.indices:
            i = int(i)
            lh_factor += V_br.value[i][f]*feats[i]
            rh_factor += (V_br.value[i][f]**2) * (feats[i]**2)
        
        factor_sum += (lh_factor**2 - rh_factor)
    factor_sum = 0.5 * factor_sum
    
    preProb = b_br.value + linear_sum + factor_sum
    
    prob = 1.0 / (1 + np.exp(-preProb))
    
    return (pair[0], prob)


def SGD_update(pair):
    """
        Args:
            pair - records are in (label, sparse feature set) format
        Broadcast:
            b - bias term (scalar)
            w - linear weight vector (array)
            k - number of factors (def=2)
            V - factor matrix of size (d dimensions, k=2 factors)
        Returns:
            gradient - ???
    """
    pass

