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
train_path = 'hdfs:/user/bm106/pub/project/cf_train.parquet'
val_path = 'hdfs:/user/bm106/pub/project/cf_validation.parquet'
test_path = 'hdfs:/user/bm106/pub/project/cf_test.parquet'
train = spark.read.parquet(train_path)
val = spark.read.parquet(val_path)
test = spark.read.parquet(test_path)

## Downsample 
user_train = set(row['user_id'] for row in train.select('user_id').distinct().collect())
user_val = set(row['user_id'] for row in val.select('user_id').distinct().collect())
user_test = set(row['user_id'] for row in test.select('user_id').distinct().collect())

user_prev = list(user_train - user_val - user_test)
k = int(0.2 * len(user_prev))

## len(user_prev) = 1019318
user_prev_filtered = random.sample(user_prev, k)
## len(user_prev_filtered) = 203863
train = train.where(train.user_id.isin(user_prev_filtered + list(user_val) + list(user_test)))

indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_indexed")
indexer_user_model = indexer_user.fit(train)
indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_indexed", handleInvalid='skip')
indexer_track_model = indexer_track.fit(train)

train = indexer_user_model.transform(train)
train = indexer_track_model.transform(train)

val = indexer_user_model.transform(val)
val = indexer_track_model.transform(val)

param = [10, 1, 10]
log_comp = True
drop_low = True
drop_thr = 2

if log_comp == True:
    train = train.select('*', F.log1p('count').alias('count_log1p'))
    val = val.select('*', F.log1p('count').alias('count_log1p'))
    rateCol = "count_log1p"
else:
    rateCol = "count"

if drop_low == True:
    train = train.filter(train['count']>drop_thr)
    val = val.filter(val['count']>drop_thr)

als = ALS(rank = param[0], maxIter=5,     regParam=param[1], userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol=rateCol, implicitPrefs=True, \
            alpha=param[2], nonnegative=True, coldStartStrategy="drop")
model = als.fit(train)

#############################################
# nmslib implementation
import nmslib
index = nmslib.init(method='hnsw', space='cosinesimil')


track_factor = model.itemFactors.toPandas()
track_vec = np.array(list(track_factor['features']))

user_factor = model.userFactors.toPandas()
#user_vec = np.array(list(user_factor['features']))


"""
user_vec = model.userFactors
user_vec = np.array(user_vec.collect())  # TODO 
track_vec = model.itemFactors
track_vecty = np.array(track_vec.collect()) # TODO 
"""

index.addDataPointBatch(track_vec)
index.createIndex({'post': 2}, print_progress=True)

# TODO
# Select user_id from validation df and extract the features from user_factor
#user_id = val.select('user_id_indexed').distinct()

user_id = val.select(val['user_id_indexed'].alias('id')).distinct()

tmp = user_id.join(user_factor,'id','inner') # TODO
user_vec = np.array(list(tmp['features'])) # TODO


pred_label = index.knnQueryBatch(user_vec, k=500, num_threads=4)

#############################################
res = model.recommendForUserSubset(user_id,500)
pred_label = res.select('user_id_indexed','recommendations.track_id_indexed')

true_label = val.select('user_id_indexed', 'track_id_indexed')\
                        .groupBy('user_id_indexed')\
                        .agg(expr('collect_list(track_id_indexed) as true_item'))

pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id_indexed', 'inner') \
            .rdd \
            .map(lambda row: (row[1], row[2]))

metrics = RankingMetrics(pred_true_rdd)
ndcg = metrics.ndcgAt(500)
mpa = metrics.precisionAt(500)
print('log_compression: ',log_comp, 'drop_low: ', drop_low)
print(param, ndcg, mpa)

# count_log1p: [10, 1, 10] 0.13079668607836065 0.0072854000000000035
# drop low counts on train and val at count = 1: [10, 1, 10] 0.10417541275032478 0.003612229187286922
# drop low counts on train and val at count = 1 and count_log1p:[10, 1, 10] 0.10592964942409674 0.0037876640724822854
# drop low counts on train and val at count = 2 and count_log1p:[10, 1, 10] 0.09861826950601185 0.0030149210903873754