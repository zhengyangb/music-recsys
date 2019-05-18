#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:

    $ spark-submit log_comp = True/False drop_low = True/False drop_thr = 0
    train_path = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
    val_path = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
    test_path = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
'''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
import itertools as it
import random
import numpy as np

def main(spark, log_comp = False, drop_low = False, drop_thr = 0):
    '''

    Parameters
    ----------
    spark : SparkSession object

    train_path : string, path to the training parquet file to load

    val_path : string, path to the validation parquet file to load

    test_path : string, path to the validation parquet file to load
    '''
    ## Load in datasets
    train_path = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
    val_path = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
    test_path = 'hdfs:/user/bm106/pub/project/cf_test.parquet'

    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)
    test = spark.read.parquet(test_path)

    ## Downsample the data
    # Pick out user list in training set
    user_train = set(row['user_id'] for row in train.select('user_id').distinct().collect())  
    # Pick out user list in validation set
    user_val = set(row['user_id'] for row in val.select('user_id').distinct().collect())
    # Get the previous 1M users
    user_prev = list(user_train - user_val)
    # Random sampling to get 20%
    k = int(0.2 * len(user_prev))
    user_prev_filtered = random.sample(user_prev, k)
    train = train.where(train.user_id.isin(user_prev_filtered + list(user_val)))

    ## Create StringIndexer
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_indexed", handleInvalid = 'skip')
    indexer_user_model = indexer_user.fit(train)
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_indexed", handleInvalid='skip')
    indexer_track_model = indexer_track.fit(train)

    train = indexer_user_model.transform(train)
    train = indexer_track_model.transform(train)

    val = indexer_user_model.transform(val)
    val = indexer_track_model.transform(val)

    test = indexer_user_model.transform(test)
    test = indexer_track_model.transform(test)

    ## ALS model
    rank_  = [5, 10, 20]
    regParam_ = [0.1, 1, 10]
    alpha_ = [1, 5, 10]
    param_grid = it.product(rank_, regParam_, alpha_)

    ## Pick out users from validation set
    user_id = val.select('user_id_indexed').distinct()
    true_label = val.select('user_id_indexed', 'track_id_indexed')\
                    .groupBy('user_id_indexed')\
                    .agg(expr('collect_list(track_id_indexed) as true_item'))

    ## Log-Compression
    ## count -> log(1+count)
    if log_comp == True:
        train = train.select('*', F.log1p('count').alias('count_log1p'))
        val = val.select('*', F.log1p('count').alias('count_log1p'))
        rateCol = "count_log1p"
    else:
        rateCol = "count"

    ## Drop interactions that have counts lower than specified threhold
    if drop_low == True:
        train = train.filter(train['count']>drop_thr)
        val = val.filter(val['count']>drop_thr)

    for i in param_grid:
        print('Start Training for {}'.format(i))
        als = ALS(rank = i[0], maxIter=10, regParam=i[1], userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol=rateCol, implicitPrefs=True, \
            alpha=i[2], nonnegative=True, coldStartStrategy="drop")
        model = als.fit(train)
        print('Finish Training for {}'.format(i))

        # Make top 500 recommendations for users in validation test
        res = model.recommendForUserSubset(user_id,500)
        pred_label = res.select('user_id_indexed','recommendations.track_id_indexed')

        pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id_indexed', 'inner') \
                    .rdd \
                    .map(lambda row: (row[1], row[2]))

        print('Start Evaluating for {}'.format(i))
        metrics = RankingMetrics(pred_true_rdd)
        map_ = metrics.meanAveragePrecision
        ndcg = metrics.ndcgAt(500)
        mpa = metrics.precisionAt(500)
        print(i, 'map score: ', map_, 'ndcg score: ', ndcg, 'map score: ', mpa)

    pass


# Only enter this block if we're in main
if __name__ == "__main__":

    ## Configurations
    # Create the spark session object
    # conf = SparkConf()
    # conf.set("spark.executor.memory", "16G")
    # conf.set("spark.driver.memory", '16G')
    # conf.set("spark.executor.cores", "4")
    # conf.set('spark.executor.instances','10')
    # conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    # conf.set("spark.default.parallelism", "40")
    # conf.set("spark.sql.shuffle.partitions", "40")
    # spark = SparkSession.builder.config(conf=conf).appName('RecomSys').getOrCreate()
    spark = SparkSession.builder.appName('RecomSys').getOrCreate()
    # Get the filename from the command line
    log_comp = sys.argv[1] 
    drop_low = sys.argv[2] 
    drop_thr = sys.argv[3]

    # Call our main routine
    main(spark, log_comp, drop_low, drop_thr)
