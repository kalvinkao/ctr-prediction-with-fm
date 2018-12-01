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
    
    # start with linear weight dot product
    linear_sum = 0.0
    for i in pair[1].indices:
        linear_sum += w[i]*pair[1].values[i]

    # factor matrix interaction sum
    factor_sum = 0.0
    lh_factor = 0.0
    rh_factor = 0.0
    
    for f in range(0, k):
        
        for i in pair[1].indices:
            lh_factor += V[i][f]*pair[1].values[i]
            rh_factor += (V[i][f]**2) * (pair[1].values[i]**2)
        
        factor_sum += (lh_factor**2 - rh_factor)
    factor_sum = 0.5 * factor_sum
    
    pred = b + linear_sum + factor_sum
    
    return ([pair[0], pred], pair[1])



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