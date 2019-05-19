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
import pandas as pd

import nmslib
import numpy as np


train_path = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
val_path = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
train = spark.read.parquet(train_path)
val = spark.read.parquet(val_path)

indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_indexed")
indexer_user_model = indexer_user.fit(train)
indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_indexed", handleInvalid='skip')
indexer_track_model = indexer_track.fit(train)

train = indexer_user_model.transform(train)
train = indexer_track_model.transform(train)

val = indexer_user_model.transform(val)
val = indexer_track_model.transform(val)


#'hdfs:/user/cy1355/log1p-10-1-10'
model_path = 'hdfs:/user/zb612/example_model.ml'
model = ALSModel.load('hdfs:/user/zb612/example_model.ml')


user_factor = model.userFactors#.toPandas()
val_user_id = val.select(val['user_id_indexed'].alias('id')).distinct()

#val_id = val_user_id.collect()
val_tmp = val_user_id.join(user_factor,'id','inner')
#val_tmp = val_tmp.toPandas() # TODO
#a = val_tmp.select('features').groupBy('id')
val_fea = val_tmp.select('features').toPandas()#collect()
val_user_vec = np.array(list(val_fea.features)) # TODO

index = nmslib.init(method='hnsw', space='cosinesimil')
track_factor = model.itemFactors.toPandas()
track_vec = np.array(list(track_factor['features']))
index.addDataPointBatch(track_vec)
index.createIndex({'post': 2}, print_progress=True)

pred_label = index.knnQueryBatch(val_user_vec, k=500, num_threads=4)

#tmp = pred_label.toPandas()
pred_tmp = np.array(pred_label)[:,0]
#import pandas as pd
#val_id = val_tmp.select('id').toPandas()

#a = val_user_id.toPandas()

# TODO
pred_label = pd.DataFrame(pred_tmp,index = val_user_id) # TODO

