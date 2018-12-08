#!/usr/bin/env python
"""
Load datasets to cluster
Transform data into wide sparse feature set
Train model and store weights and loss to file
Evaluate loss on a validation set (with labels)

[Make predictions on test.txt and save to file? DO LAST]
"""

###########################################################################################################################################
#Prep 
##########################################################################################################################################
# import packages here
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import DataFrame

# start Spark Session
from pyspark.sql import SparkSession
app_name = "w261FinalProject"
master = "local[*]"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .master(master)\
        .getOrCreate()
sc = spark.sparkContext

# define working directories
#dataFolder = "gs://w261_jpalcckk/data/"
dataFolder = "data/"
#resultsFolder = "gs://w261_jpalcckk/results/"
resultsFolder = "results/"

# load .txt data here
# will look something like this:
#trainRDD = sc.textFile('gs://w261bucket/train.txt')
fullRDD = sc.textFile(dataFolder+'train.txt')

#break the trainfile into pieces to have a holdout set
excludeRDD, TestRDD, TrainRDD = fullRDD.randomSplit([0.999, 0.0005, 0.0005], seed = 1)
TrainRDD.cache()
TestRDD.cache()

# function to parse raw data and tag feature values with type and feature indices
def parseCV(line):
    """
    Map record_csv_string --> (label, features)
    """

    # start of categorical features
    col_start = 14
    
    raw_values = line.split('\t')
    label = int(raw_values[0])
    
    # parse numeric features
    numericals = []
    for idx, value in enumerate(raw_values[1:col_start]):
        if value == '':
            append_val = 'NA'
        elif value == '0':
            append_val = '0'
        else:
            # continues variables
            if idx in [0,3,6,7]:
                if float(value)<10:
                    append_val = '<10'
                elif float(value)<25:
                    append_val = '<25'
                else:
                    append_val = '>25'
            elif idx in [1,2,5]:
                if float(value)<100:
                    append_val = '<100'
                else:
                    append_val = '>100'
            elif idx==4:
                if float(value)<10000:
                    append_val = '<10k'
                elif float(value)<50000:
                    append_val = '<50k'
                else:
                    append_val = '>50k'
            elif idx==8:
                if float(value)<100:
                    append_val = '<100'
                elif float(value)<500:
                    append_val = '<500'
                else:
                    append_val = '>500'
            elif idx in [10,11]:
                if float(value)<3:
                    append_val = '<3'
                elif float(value)<6:
                    append_val = '<6'
                else:
                    append_val = '>6'
            elif idx==12:
                if float(value)<5:
                    append_val = '<5'
                elif float(value)<10:
                    append_val = '<10'
                elif float(value)<25:
                    append_val = '<25'
                else:
                    append_val = '>25'
            # ordinal/binary cases
            else:
                append_val = str(value)
                
        numericals.append('n' + str(idx) + '_' + append_val)
            
    # parse categorical features
    categories = []
    for idx, value in enumerate(raw_values[col_start:]):
        if value == '':
            categories.append('c'+ str(idx) + '_NA')
        else:
            categories.append('c'+ str(idx) + '_' + str(value))

    return Row(label=label, raw=numericals + categories)

# function to one hot encode all features using a count vectorizer
def vectorizeCV(DF):
    
    vectorizer = CountVectorizer()
    cv = CountVectorizer(minDF=.0001, inputCol="raw", outputCol="features")
    
    model = cv.fit(DF)
    result = model.transform(DF)
    
    return result, model

# call functions
parsedDF = TrainRDD.map(parseCV).toDF().cache()
vectorizedDF, cvModel = vectorizeCV(parsedDF)
#cvModel.save("cvModel")

#convert back to RDDs
vectorizedRDD = vectorizedDF.select(['label', 'features']).rdd.cache()

num_feats = vectorizedRDD.take(1)[0][1].size
#percent_pos = vectorizedRDD.map(lambda x: x[0]).mean()

############################################################################################################################################Build Model & store losses
##########################################################################################################################################
# build parsed train data, train model, 
def predict_grad(pair, k_br, b_br, w_br, V_br):
    """
        Compute the predicted probability AND return the gradients
        Args:
            pair - records are in (label, sparse feature set) format
        Broadcast:
            b - bias term (scalar)
            w - linear weight vector (array)
            k - number of factors (def=2)
            V - factor matrix of size (d dimensions, k=2 factors)
        Returns:
            predRDD - pair of ([label, predicted probability], [set of weight vectors in csr_matrix format])
    """
    
    label = pair[0]
    feats = pair[1]
    
    # start with linear weight dot product
    linear_sum = np.dot(w_br.value[0][feats.indices], feats.values)

    # factor matrix interaction sum
    factor_sum = 0.0
    lh_factor = [0.0]*k_br.value
    rh_factor = [0.0]*k_br.value
    
    for f in range(0, k_br.value):
        lh_factor[f] = np.dot(V_br.value[f][feats.indices], feats.values)  #KEY--this is used in v_grad matrix below
        rh_factor[f] = np.dot(V_br.value[f][feats.indices]**2, feats.values**2)
        factor_sum += (lh_factor[f]**2 - rh_factor[f])
    factor_sum = 0.5 * factor_sum
    
    pre_prob = b_br.value + linear_sum + factor_sum
    
    prob = 1.0 / (1 + np.exp(-pre_prob))  #logit transformation
    
    #compute Gradients
    b_grad = prob - label
    
    w_grad = csr_matrix((b_grad*feats.values, (np.zeros(feats.indices.size), feats.indices)), shape=(1, w_br.value.shape[1]))
    
    # V matrix
    v_data = np.array([], dtype=np.float32)
    v_rows = np.array([], dtype=int)
    v_cols = np.array([], dtype=int)
    for i in range(0, k_br.value):
        v_data = np.append(v_data, b_grad*(lh_factor[i]*feats.values - np.multiply(V_br.value[i][feats.indices], feats.values**2)))
        v_rows = np.append(v_rows, [i]*feats.indices.size)
        v_cols = np.append(v_cols, feats.indices)
    v_grad = csr_matrix((v_data, (v_rows, v_cols)), shape=(k_br.value, V_br.value.shape[1]))
    
    return ([label, prob], [b_grad, w_grad, v_grad])

def reduceFct(x, y):
    """function for summing bias and weight matrices
        arguments: ([label, pred], [bias, weight, V matrix])
        out:       [sum bias b, sum weight w, sum matrix V]
    """
    b = x[0] + y[0]
    w = x[1] + y[1]
    V = x[2] + y[2]
    return [b, w, V]

def logLoss(pair):
    """parallelize log loss
        input: ([label, prob], [b_grad, w_grad, v_grad])
    """
    y = pair[0][1]
    
    eps = 1.0e-16
    if pair[0][1] == 0:
        y_hat = eps
    elif pair[0][1] == 1:
        y_hat = 1-eps
    else:
        y_hat = pair[0][1]
    
    return -(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

def iterateSGD(dataRDD, k, bInit, wInit, vInit, nIter = 2, learningRate = 0.1, useReg = False, regParam = 0.001):
    """iterate over vectorized RDD to update weight vectors and bias with option of regularization"""
    k_br = sc.broadcast(k)    
    b_br = sc.broadcast(bInit)
    w_br = sc.broadcast(wInit)
    V_br = sc.broadcast(vInit)

    losses = []
    N = dataRDD.count()

    for i in range(0, nIter):
        print('-' * 25 + 'Iteration ' + str(i) + '-' * 25)
        predRDD = dataRDD.map(lambda x: predict_grad(x, k_br, b_br, w_br, V_br)).cache()
        #print(predRDD.take(1))
        
        loss = predRDD.map(logLoss).reduce(lambda a,b: a+b)/N + \
                int(useReg)*(regParam/2)*(np.linalg.norm(w_br.value)**2 + np.linalg.norm(V_br.value)**2)
        losses.append(loss)
        print(f'Current log-loss: {loss}')
        
        # NEW reduce step
        gradRDD = predRDD.values().reduce(reduceFct)
        bGrad = gradRDD[0]/N
        wGrad = gradRDD[1]/N
        vGrad = gradRDD[2]/N

        print(f"Bias: {bGrad}")
        print(f"wGrad shape: {wGrad.shape}")
        print(f"vGrad shape: {vGrad.shape}")

        ############## update weights ##############
        # first, unpersist broadcasts
        b_br.unpersist()
        w_br.unpersist()
        V_br.unpersist()

        # update
        b_br = sc.broadcast(b_br.value - learningRate * bGrad)
        w_br = sc.broadcast(w_br.value - learningRate * (wGrad.toarray()+int(useReg)*regParam*np.linalg.norm(w_br.value)))
        V_br = sc.broadcast(V_br.value - learningRate * (vGrad.toarray()+int(useReg)*regParam*np.linalg.norm(V_br.value)))
        
    return losses, b_br, w_br, V_br

# initialize weights
np.random.seed(24)
k = 2
b = 0.0
w = np.random.normal(0.0, 0.02, (1, num_feats))
V = np.random.normal(0.0, 0.02, (k, num_feats))

nIter = 2
start = time.time()
losses, b_br, w_br, V_br = iterateSGD(vectorizedRDD, k, b, w, V, nIter, learningRate = 0.1, useReg = True)
print(f'Performed {nIter} iterations in {time.time() - start} seconds')

##########################################################################################################################################
#write weights to file
##########################################################################################################################################
np.savetxt(resultsFolder+"w_weights.txt", w_br.value, delimiter=',')
np.savetxt(resultsFolder+"V_weights.txt", V_br.value, delimiter=',')

file = open(resultsFolder+"beta.txt", "w")
file.write(str(b_br.value))
file.close()

with open(resultsFolder+'train_loss.txt', 'w') as f:
    for item in losses:
        f.write("%s\t" % item)
##########################################################################################################################################
#make predictions on holdout (labeled) set
##########################################################################################################################################
parsedTestDF = TestRDD.map(parseCV).toDF().cache()
vectorizedTestDF = cvModel.transform(parsedTestDF)
vectorizedTestRDD = vectorizedTestDF.select(['label', 'features']).rdd.cache()

def predict_prob(pair, k_br, b_br, w_br, V_br):
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
            predRDD - pair of (label, predicted probability)
    """
    
    label = pair[0]
    feats = pair[1]
    
    # start with linear weight dot product
    linear_sum = np.dot(w_br.value[0][feats.indices], feats.values)

    # factor matrix interaction sum
    factor_sum = 0.0
    lh_factor = [0.0]*k_br.value
    rh_factor = [0.0]*k_br.value
    
    for f in range(0, k_br.value):
        lh_factor[f] = np.dot(V_br.value[f][feats.indices], feats.values)
        rh_factor[f] = np.dot(V_br.value[f][feats.indices]**2, feats.values**2)
        factor_sum += (lh_factor[f]**2 - rh_factor[f])
    factor_sum = 0.5 * factor_sum
    
    pre_prob = b_br.value + linear_sum + factor_sum
    
    prob = 1.0 / (1 + np.exp(-pre_prob))  #logit transformation
    
    return (label, prob)

def testLoss(pair):
    """parallelize log loss
        input: (label, prob)
    """
    y = pair[0]
    
    eps = 1.0e-16
    if pair[1] == 0:
        y_hat = eps
    elif pair[1] == 1:
        y_hat = 1-eps
    else:
        y_hat = pair[1]
    
    return -(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

k_br = sc.broadcast(k)

testLoss = vectorizedTestRDD.map(lambda x: predict_prob(x, k_br, b_br, w_br, V_br)) \
                            .map(testLoss).mean()

print("Log-loss on the hold-out test set is:", testLoss)

# save test loss to file
with open(resultsFolder+'test_loss.txt', 'w') as f:
    f.write(str(testLoss))

##########################################################################################################################################
#make predictions on unlabeled 'test.txt' dataset
##########################################################################################################################################

unlabeledRDD = sc.textFile(dataFolder+'test.txt')
largeUnlabeledRDD, smallUnlabeledRDD = unlabeledRDD.randomSplit([0.999, 0.001], seed = 1)
smallUnlabeledRDD.cache()

parsedUnlabeledDF = smallUnlabeledRDD.map(lambda x: "0\t"+x).map(parseCV).toDF().cache()
vectorUnlabeledDF = cvModel.transform(parsedUnlabeledDF)

vectorUnlabeledRDD = vectorUnlabeledDF.select(['raw','features']).rdd.cache()

unlabeledPred = vectorUnlabeledRDD.map(lambda x: predict_prob(x, k_br, b_br, w_br, V_br)).cache()

predsTimeStamp = str(int(np.floor(time.time())))
unlabeledPred.coalesce(1,True).saveAsTextFile(resultsFolder+"test_predictions_"+predsTimeStamp)

