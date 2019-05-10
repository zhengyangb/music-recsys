#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:

    $ spark-submit train_path val_path
    train_path = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
    val_path = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
'''

# nohup spark-submit --conf spark.driver.memory=16g --conf spark.executor.memory=16g --conf spark.sql.shuffle.partitions=40 draft.py hdfs:/user/bm106/pub/project/cf_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet &>final.log&
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
def main(spark, train_path, val_path):
    '''Main routine for unsupervised training

    Parameters
    ----------
    spark : SparkSession object

    train_path : string, path to the training parquet file to load

    val_path : string, path to the validation parquet file to load
    '''
    
    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)

    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_indexed")
    indexer_user_model = indexer_user.fit(train)
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_indexed")
    indexer_track_model = indexer_track.fit(train)

    train = indexer_user_model.transform(train)
    train = indexer_track_model.transform(train)

    val = indexer_user_model.transform(val)
    val = indexer_track_model.transform(val)

    # ALS model
    rank_  = [5,10,20]
    regParam_ = [0.1, 1,10]
    alpha_ = [10, 20, 40]
    param_grid = it.product(rank_, regParam_, alpha_)
    ndcg_list = []
    mpa_list = []
    for i in param_grid:
        print('Start Training for {}'.format(i))
        als = ALS(rank = i[0], maxIter=5, regParam=i[1], userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol="count", implicitPrefs=True, \
            alpha=i[2], nonnegative=True, coldStartStrategy="drop")
        model = als.fit(train)
        print('Finish Training for {}'.format(i))

        predictions = model.transform(val)

        pred_label = predictions.select('user_id_indexed', 'track_id_indexed', 'prediction')\
                                .groupBy('user_id_indexed')\
                                .agg(expr('collect_list(track_id_indexed) as pred_item'))

        true_label = val.select('user_id_indexed', 'track_id_indexed')\
                                .groupBy('user_id_indexed')\
                                .agg(expr('collect_list(track_id_indexed) as true_item'))

        pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id_indexed', 'inner') \
                    .rdd \
                    .map(lambda row: (row[1], row[2]))

        print('Start Evaluating for {}'.format(i))
        metrics = RankingMetrics(pred_true_rdd)
        ndcg = metrics.ndcgAt(500)
        mpa = metrics.precisionAt(500)
        ndcg_list.append(ndcg)
        mpa_list.append(mpa)
        print(i, ndcg, mpa)

    pass


# Only enter this block if we're in main
if __name__ == "__main__":

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
    train_path = sys.argv[1]

    # And the location to store the trained model
    val_path = sys.argv[2]

    # Call our main routine
    main(spark, train_path, val_path)
