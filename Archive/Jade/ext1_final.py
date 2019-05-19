from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
import itertools as it
import numpy as np
import pandas as pd

train_path = 'hdfs:/user/cy1355/train_pre_noindex.parquet'
#train_path = 'hdfs:/user/cy1355/train_pre.parquet'
#val_path = 'hdfs:/user/cy1355/val_pre.parquet'

#train_path = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
val_path = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
test_path = 'hdfs:/user/bm106/pub/project/cf_test.parquet'

train = spark.read.parquet(train_path)
val = spark.read.parquet(val_path)
test = spark.read.parquet(test_path)

indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_indexed",handleInvalid='skip')
indexer_user_model = indexer_user.fit(train)
indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_indexed", handleInvalid='skip')
indexer_track_model = indexer_track.fit(train)

train = indexer_user_model.transform(train)
train = indexer_track_model.transform(train)

val = indexer_user_model.transform(val)
val = indexer_track_model.transform(val)

test = indexer_user_model.transform(test)
test = indexer_track_model.transform(test)

# Best hyperparameter: rank = 20, reg = 0.1, alpha = 10
param = [20, 0.1, 10]

# Set the strategies
log_comp = True
drop_low = False
drop_thr = 2

if log_comp == True:
    train = train.select('*', F.log1p('count').alias('count_log1p'))
    val = val.select('*', F.log1p('count').alias('count_log1p'))
    test = test.select('*', F.log1p('count').alias('count_log1p'))
    rateCol = "count_log1p"
else:
    rateCol = "count"

if drop_low == True:
    train = train.filter(train['count']>drop_thr)
    #val = val.filter(val['count']>drop_thr)

als = ALS(rank = param[0], maxIter=10, regParam=param[1], userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol=rateCol, implicitPrefs=True, \
            alpha=param[2], nonnegative=True, coldStartStrategy="drop")
model = als.fit(train)

#model_path = 'log1p_' + str(param[0]) +'_' + str(param[1])+ '_' + str(param[2])
#save_path = 'hdfs:/user/jl9875/als_model_'
#model.save(save_path + model_path)
#model.write().overwrite().save(save_path + model_path)


# Load pre-trained model
#model = ALSModel.load(save_path + model_path)

user_id = val.select('user_id_indexed').distinct()
res = model.recommendForUserSubset(user_id,500)
pred_label = res.select('user_id_indexed','recommendations.track_id_indexed')

true_label = val.select('user_id_indexed', 'track_id_indexed')\
                        .groupBy('user_id_indexed')\
                        .agg(expr('collect_list(track_id_indexed) as true_item'))

pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id_indexed', 'inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))

metrics = RankingMetrics(pred_true_rdd)
#ndcg = metrics.ndcgAt(500)
mpa = metrics.precisionAt(500)
MAP = metrics.meanAveragePrecision
print('log_compression:',log_comp, 'drop_low:', drop_low, 'drop_thr:', drop_thr, 'param:',param, 'mpa:', mpa,'MAP:', MAP)


# Load pre-trained model
#model = ALSModel.load('hdfs:/user/jl9875/als_model_log1p_20_0.1_10')

# The best performance is given by log compression strategy only
# Predictin on test

test_user_id = test.select('user_id_indexed').distinct()
test_res = model.recommendForUserSubset(test_user_id,500)
test_pred_label = test_res.select('user_id_indexed','recommendations.track_id_indexed')

test_true_label = test.select('user_id_indexed', 'track_id_indexed')\
                        .groupBy('user_id_indexed')\
                        .agg(expr('collect_list(track_id_indexed) as true_item'))

test_pred_true_rdd = test_pred_label.join(F.broadcast(test_true_label), 'user_id_indexed', 'inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))

test_metrics = RankingMetrics(test_pred_true_rdd)
#ndcg = metrics.ndcgAt(500)
test_mpa = test_metrics.precisionAt(500)
test_MAP = test_metrics.meanAveragePrecision
test_ndcg = test_metrics.ndcgAt(500)
print('Prediction on test')
#print('log_compression:',log_comp, 'drop_low:', drop_low, 'drop_thr:', drop_thr, 'param:',param, 'mpa:', mpa,'MAP:', MAP)
print('test_mpa:', test_mpa,'test_MAP:', test_MAP, 'test_ndcg', test_ndcg)

