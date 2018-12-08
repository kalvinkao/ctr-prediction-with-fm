#!/usr/bin/env python
"""
Load datasets to cluster
Transform data into wide sparse feature set
Train model and store weights and loss to file
Evaluate loss on a validation set (with labels)

[Make predictions on test.txt and save to file? DO LAST]
"""


# import packages here



# load .txt data here
# will look something like this:
trainRDD = sc.textFile('gs://w261bucket/train.txt')


# build parsed train data, train model, store losses, write weights to file, make predictions on holdout (labeled) set