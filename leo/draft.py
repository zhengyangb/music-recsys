#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Part 1: unsupervised model training

Usage:

    $ spark-submit unsupervised_train.py hdfs:/path/to/file.parquet hdfs:/path/to/save/model

'''


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

# TODO: you may need to add imports here
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline

def main(spark, data_file, model_file):
    '''Main routine for unsupervised training

    Parameters
    ----------
    spark : SparkSession object

    data_file : string, path to the parquet file to load

    model_file : string, path to store the serialized model file
    '''

    dataset = spark.read.parquet(data_file)

    # Select out the 20 attribute columns labeled mfcc_00, mfcc_01, ..., mfcc_19
    idx_num = [i for i in range(20)]
    input_col = list(map(select,idx_num))

    assembler = VectorAssembler(inputCols=input_col, outputCol="features")

    # Normalize the features using a StandardScaler
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

    # Fit a K-means clustering model to the standardized data with K=100.
    kmeans = KMeans(featuresCol='scaledFeatures').setK(100).setSeed(1)

    # create pipeline
    pipeline = Pipeline(stages = [assembler, scaler, kmeans])
    kmeans_model = pipeline.fit(dataset)

    # save the model
    kmeans_model.save(model_file)
    pass


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('unsupervised_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    # Call our main routine
    main(spark, data_file, model_file)
