# Databricks notebook source
# MAGIC %md ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

from pyspark.sql.types import IntegerType, FloatType

# File location and type
#file_location = "/FileStore/tables/ratings.csv"
file_location = "/FileStore/tables/ratings_small.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# We drop the time stamp
df = df.drop("timestamp")

# We convert to ints
df = df.withColumn("userId", df["userId"].cast(IntegerType()))
df = df.withColumn("movieId", df["movieId"].cast(IntegerType()))
df = df.withColumn("rating", df["rating"].cast(FloatType()))

display(df)

# COMMAND ----------

# First we compute the unique users

unique_usr = df.select('userId').distinct().collect()
unique_usr = [row.asDict()["userId"] for row in unique_usr]

usr_to_emb = {usr : i for i,usr in enumerate(unique_usr)}
emb_to_usr = {i : usr for i,usr in enumerate(unique_usr)}

# COMMAND ----------

unique_movie = df.select('movieId').distinct().collect()
unique_movie = [int(row.asDict()["movieId"]) for row in unique_movie]

movie_to_emb = {movie : i for i,movie in enumerate(unique_movie)}
emb_to_movie = {i : movie for i,movie in enumerate(unique_movie)}


# COMMAND ----------

from pyspark.sql.functions import col, create_map, lit
from itertools import chain

mapping_movie = create_map([lit(x) for x in chain(*movie_to_emb.items())])
mapping_usr = create_map([lit(x) for x in chain(*usr_to_emb.items())])

df = df.withColumn("movieId", mapping_movie.getItem(col("movieId")))
df = df.withColumn("userId", mapping_usr.getItem(col("userId")))

# COMMAND ----------

# TO DO : take care of the JOIN and the Cartesian product

# COMMAND ----------

t = keys.partitionBy(N_WORKERS).glom().collect()

# COMMAND ----------

keys = y.keys()
keys = keys.map(lambda x : (x[1],1))
keys = keys.reduceByKey(lambda x,y : x + y)

# COMMAND ----------

t = keys.take(10)

# COMMAND ----------

from pyspark.sql.types import StructType, IntegerType, StructField

schema = StructType([StructField("movieId", IntegerType(), True), StructField("count", IntegerType(), True),])

# COMMAND ----------

t = sqlContext.createDataFrame(keys,schema)

# COMMAND ----------

from pyspark.sql.window import *
window = Window.partitionBy("movieId")
df = df.withColumn("CumSumTotal", sum(df.count).over(window))

# COMMAND ----------

df.createOrReplaceTempView("test")

# COMMAND ----------

movie_to_count = sqlContext.sql("select movieId,count, " +
  "SUM(count) over (  order by movieId  rows between unbounded preceding and current row ) cum_count " +
  "from test").rdd.map(lambda x : (x["movieId"], x["cum_count"])).map(lambda x : x[1]).collect()

# COMMAND ----------

N = y.count()

# COMMAND ----------

N

# COMMAND ----------

def count_partition(key):
  key = int(( float(movie_to_count[key]) / N) * N_WORKERS)
  return(key)

# COMMAND ----------

movie_to_count[]

# COMMAND ----------

test = y.keys().map(lambda x : (x[1],x[0]))


# COMMAND ----------

t = test.partitionBy(N_WORKERS,count_partition)

# COMMAND ----------

# Reverse key so that movieId is first
t = y.map(lambda x : ( x[0][1], (x[0][0] , x[1] ) ) )

# COMMAND ----------

t.take(10)

# COMMAND ----------



# COMMAND ----------

t.partitionBy(N_WORKERS,count_partition)

# COMMAND ----------

y = df.rdd.map(lambda x : ( (x["userId"],x["movieId"]), int(x["rating"])) )

# COMMAND ----------



# COMMAND ----------

y.map(lambda x : ((x[0][1],x[0][0]),x[1] ).take(10)

# COMMAND ----------

for 

# COMMAND ----------

import numpy as np

# HHYPER-PARAMETERS

N_EPOCH = 100
step = 0.01
hidden_size = 5

n = len(usr_to_emb)
m = len(movie_to_emb)

# Matrixes which represent our embeddings
U = np.random.randn(n,hidden_size)
V = np.random.randn(m,hidden_size)

# We then create our RDDs from the matrixes
u_rows = sc.parallelize((i,U[i]) for i in range(n))
v_rows = sc.parallelize((i,V[i]) for i in range(m))

# We create an RDD with key = (i,j) and values = rating
y = df.rdd.map(lambda x : ( (x["userId"],x["movieId"]), int(x["rating"])) )

# We then broadcast it
sc.broadcast(y)

for l in range(N_EPOCH):
  
  # We get the cartesian product 
  u_v = u_rows.cartesian(v_rows)

  # Then we get a key value with keys = (i,j), and values = (u,v) i.e. the index of user/product and the associated vectors
  u_v_vect = u_v.map(lambda x : ( (x[0][0],x[1][0]) , (x[0][1],x[1][1]) ) ) 
  u_v_vect.persist()

  u_v_dot =  u_v.map(lambda x : ( (x[0][0],x[1][0]) , np.dot(x[0][1],x[1][1]) ) )
  u_v_dot.persist()
  
  # We keep only the vectors where there is an observation and keep 
  u_v_y_dot = y.join(u_v_dot)
  
  # Free memory from the cartesian product
  u_v.unpersist()
  
  # We then take the difference between the prediction and the real value
  diff = u_v_y_dot.mapValues(lambda x : (x[1] - x[0]))
  diff.persist()

  # We compute the mean squared loss
  loss = diff.mapValues(lambda x : x**2).map(lambda x: x[1]).mean() / 2
  
  # We print the loss
  print("Current loss at epoch %i : %f" % (l,loss))
  
  # We compute the gradients for each pair
  grads = u_v_vect.join(diff)
  grads.persist()

  # We then compute the gradients wrt u and v
  grad_u = grads.map(lambda x : (x[0][0], x[1][0][1] * x[1][1]))
  grad_v = grads.map(lambda x : (x[0][1], x[1][0][0] * x[1][1]))
  
  # We take the average of each batch
  agg_grad_u = grad_u.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  agg_grad_v = grad_v.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  
  # We make one step of a gradient descent
  u_rows = u_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1])
  v_rows = v_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1])
  

# COMMAND ----------

y = y.collect()

# COMMAND ----------

sc.broadcast(y)

# COMMAND ----------

import numpy as np

# HHYPER-PARAMETERS

N_EPOCH = 100
step = 0.01
hidden_size = 5

n = len(usr_to_emb)
m = len(movie_to_emb)

# Matrixes which represent our embeddings
U = np.random.randn(n,hidden_size)
V = np.random.randn(m,hidden_size)

# We then create our RDDs from the matrixes
u_rows = sc.parallelize((i,U[i]) for i in range(n))
v_rows = sc.parallelize((i,V[i]) for i in range(m))

v_rows_broad = sc.broadcast(v_rows.collect())

# COMMAND ----------



# COMMAND ----------

""" coalesce : merger localement
 broadcaster 
 unperist
 
 Checkpoint : toutes les 50 opérations
 Cocoa : API Spark pour updater localement
 
 """

# COMMAND ----------

print("The current loss is %f" % loss)

# COMMAND ----------

y.count()

# COMMAND ----------

len(y.keys().collect())

# COMMAND ----------

n

# COMMAND ----------

n*m

# COMMAND ----------

len(u_v_dot.keys().collect())

# COMMAND ----------

372100 - 100836

# COMMAND ----------

u_v_y_dot.count()

# COMMAND ----------

diff.take(1)

# COMMAND ----------



# COMMAND ----------

diff

# COMMAND ----------



# COMMAND ----------

step = 0.01



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

u_rows.take(10)

# COMMAND ----------

t.take(10)

# COMMAND ----------

sorted(agg_grad_u.keys().collect())

# COMMAND ----------

t = u_rows.leftOuterJoin(agg_grad_u)

# COMMAND ----------

t

# COMMAND ----------

t[1][0][1]

# COMMAND ----------

t[0][0]

# COMMAND ----------

diff.take(10)

# COMMAND ----------

t = grad_u.take(10)[0]

# COMMAND ----------

t[1][0]

# COMMAND ----------


t[1][0][1] * t[1][1]

# COMMAND ----------

grad_u = u_v_vect.join(diff)
grad_u = grad_u.map(lambda x : (x[0][0], x[1][0][1] * x[1][1]))

# COMMAND ----------

grad_u.groupByKey()

# COMMAND ----------

rdd = rdd.map(lambda x : (x["userId"],x["movieId"]))

# COMMAND ----------

rdd = rdd.partitionBy(8)

# COMMAND ----------



# COMMAND ----------

df.dtypes

# COMMAND ----------

t = df.select(["movieId","userId"]).rdd.map(tuple)

# COMMAND ----------

def acc_grad(x,y):
  u

t.groupByKey().reduceByKey()

# COMMAND ----------

import pickle

# Saving the mappings

with open('emb_to_movie.pickle', 'wb') as handle:
    pickle.dump(emb_to_movie, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('emb_to_usr.pickle', 'wb') as handle:
    pickle.dump(emb_to_usr, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('usr_to_emb.pickle', 'wb') as handle:
    pickle.dump(usr_to_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('movie_to_emb.pickle', 'wb') as handle:
    pickle.dump(movie_to_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

# COMMAND ----------

import pickle

"""with open('./emb_to_movie.pickle', 'r') as handle:
  emb_to_movie = pickle.load(handle)"""

# COMMAND ----------

from pyspark.sql.types import IntegerType, FloatType

file_location = "/FileStore/tables/ratings_small_embedded.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

df = df.withColumn("userId", df["userId"].cast(IntegerType()))
df = df.withColumn("movieId", df["movieId"].cast(IntegerType()))
df = df.withColumn("rating", df["rating"].cast(FloatType()))

# COMMAND ----------

display(df)

# COMMAND ----------



# COMMAND ----------

import pyspark.sql.functions as F

df.agg(F.countDistinct("movieId")).collect()[0]

# COMMAND ----------

df.agg(F.countDistinct("userId")).collect()[0]

# COMMAND ----------

# Create a view or table

temp_table_name = "ratings_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------



# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "ratings_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)