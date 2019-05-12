from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
import itertools as it

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

als = ALS(rank = param[0], maxIter=5, regParam=param[1], userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol=rateCol, implicitPrefs=True, \
            alpha=param[2], nonnegative=True, coldStartStrategy="drop")
model = als.fit(train)
user_id = val.select('user_id_indexed').distinct()

#############################################
# nmslib implementation
import nmslib
index = nmslib.init(method='hnsw', space='cosinesimil')
user_vec = model.userFactors
user_vec = np.array(user_vec.collect())  # TODO 
track_vec = model.itemFactors
track_vecty = np.array(track_vec.collect()) # TODO 

index.addDataPointBatch(track_vec)
index.createIndex({'post': 2}, print_progress=True)
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