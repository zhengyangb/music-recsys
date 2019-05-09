#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:

    $ spark-submit train_path val_path
    train_path = 'hdfs:/user/zb612/transformed_train.parquet'
    val_path = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
'''

# spark-submit --conf spark.driver.memory=16g --conf spark.executor.memory=16g draft.py hdfs:/user/zb612/transformed_train.parquet hdfs:/user/bm106/pub/project/cf_validation.parquet
# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
import itertools as it
def main(spark, data_file, model_file):
    '''Main routine for unsupervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''
    # train = spark.read.parquet('hdfs:/user/bm106/pub/project/cf_train.parquet')
    # # downsample the dataset
    # train = train.sample(False, 0.1, seed=1)

    # # transform the user and item identifiers (strings) into numerical index representations
    # indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_indexed")
    # train = indexer_user.fit(train).transform(train)

    # indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_indexed")
    # train = indexer_track.fit(train).transform(train)

    # train.write.format("parquet").mode("overwrite").save('transformed_train.parquet')
    
    train = spark.read.parquet(train_path)
    val = spark.read.parquet(val_path)
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_indexed")
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_indexed")
    val = indexer_user.fit(val).transform(val)
    val = indexer_track.fit(val).transform(val)

    # ALS model
    rank_  = [5,10,20]
    regParam_ = [0.1, 1,10]
    alpha_ = [10, 20, 40]
    param_grid = it.product(rank_, regParam_, alpha_)
    ndcg_list = []
    mpa_list = []
    for i in param_grid:
        als = ALS(rank = i[0], maxIter=5, regParam=i[1], userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol="count", implicitPrefs=True, \
            alpha=i[2], nonnegative=True, coldStartStrategy="drop")
        model = als.fit(train)

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

        metrics = RankingMetrics(pred_true_rdd)
        ndcg = metrics.ndcgAt(500)
        mpa = metrics.precisionAt(500)
        ndcg_list.append(ndcg)
        mpa_list.append(mpa)
        print(param_grid[i], ndcg, mpa)

    pass


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('RecomSys').getOrCreate()

    # Get the filename from the command line
    train_path = sys.argv[1]

    # And the location to store the trained model
    val_path = sys.argv[2]

    # Call our main routine
    main(spark, train_path, val_path)
