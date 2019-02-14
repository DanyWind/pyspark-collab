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


# COMMAND ----------

# First we compute the unique users

unique_usr = df.select('userId').distinct().collect()
unique_usr = [row.asDict()["userId"] for row in unique_usr]

usr_to_emb = {usr : i for i,usr in enumerate(unique_usr)}
emb_to_usr = {i : usr for i,usr in enumerate(unique_usr)}

unique_movie = df.select('movieId').distinct().collect()
unique_movie = [int(row.asDict()["movieId"]) for row in unique_movie]

movie_to_emb = {movie : i for i,movie in enumerate(unique_movie)}
emb_to_movie = {i : movie for i,movie in enumerate(unique_movie)}

# COMMAND ----------

import numpy as np

def sigmoid(x):
  output = 1 / (1 + np.exp(-x))
  return(output)

# COMMAND ----------

""" Version 1 Vanilla """

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
sc.broadcast(y.collect())

for l in range(N_EPOCH):
  
  if l > 0:
    u_rows = sc.parallelize(u_rows)
    v_rows = sc.parallelize(v_rows)
  
  # We get the cartesian product 
  u_v = u_rows.cartesian(v_rows)

  # Then we get a key value with keys = (i,j), and values = (u,v) i.e. the index of user/product and the associated vectors
  u_v_vect = u_v.map(lambda x : ( (x[0][0],x[1][0]) , (x[0][1],x[1][1]) ) ) 
  u_v_vect.persist()

  u_v_dot =  u_v.map(lambda x : ( (x[0][0],x[1][0]) , np.dot(x[0][1],x[1][1]) ) )
  
  # We keep only the vectors where there is an observation and keep 
  u_v_y_dot = y.join(u_v_dot)
  
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
  u_rows = u_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1]).collect()
  v_rows = v_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1]).collect()
  
  # Free u_v_vect and diff
  u_v_vect.unpersist()
  diff.unpersist()
  grads.unpersist()

# COMMAND ----------

Time : 23.06 minutes
MSE : 3.239010

# COMMAND ----------

import numpy as np

# HHYPER-PARAMETERS

N_EPOCH = 25
step = 0.01
hidden_size = 5

n = len(usr_to_emb)
m = len(movie_to_emb)

# Matrixes which represent our embeddings
U = np.random.randn(n,hidden_size)
V = np.random.randn(m,hidden_size)

from pyspark.ml.linalg import Vectors

# COMMAND ----------

y = df.rdd.map(lambda x : (usr_to_emb[x["userId"]], movie_to_emb[x["movieId"]], x["rating"]))
y = y.toDF(["userId","movieId","rating"])

# COMMAND ----------

from pyspark.sql.functions import broadcast

t = mydf.join(broadcast(y), mydf.userId == y.userId).collect()

# COMMAND ----------

y = df.rdd.map(lambda x : (usr_to_emb[x["userId"]], movie_to_emb[x["movieId"]], x["rating"]))


# COMMAND ----------

a = y.join(u_rows).collect()

# COMMAND ----------

u_rows = sc.parallelize((i,U[i]) for i in range(n))

# COMMAND ----------

mydf.join(broa)

# COMMAND ----------

"""Version 2 : deleted cartesian product and optimized join"""

import numpy as np

# HHYPER-PARAMETERS

N_EPOCH = 25
step = 0.01
hidden_size = 8

n = len(usr_to_emb)
m = len(movie_to_emb)

# Matrixes which represent our embeddings
U = np.random.randn(n,hidden_size)
V = np.random.randn(m,hidden_size)

# We then create our RDDs from the matrixes
u_rows = sc.parallelize((i,U[i]) for i in range(n))
v_rows = sc.parallelize((i,V[i]) for i in range(m))

# We create an RDD with key = (i,j) and values = rating
y = df.rdd.map(lambda x : ( (x["userId"],x["movieId"]), int(x["rating"])) )

for l in range(N_EPOCH):
  
  if l > 0:
    u_rows = sc.parallelize(u_rows)
    v_rows = sc.parallelize(v_rows)
  
  # First we map the observations with the local U matrix
  a = y.map(lambda x : (x[0][0], (x[0][1] , x[1]) )).join(u_rows)
  
  # Then we reorder in order to join with the V matrix
  b = a.map(lambda x : (x[1][0][0], (x[0], x[1][1], x[1][0][1]) ))
  
  # We then join and reorder into ((i,j) ,(u,v,y,y_pred,diff))
  c = b.join(v_rows)
  u_v_y_dot = c.map(lambda x : ((x[1][0][0], x[0]) , (x[1][0][1], x[1][1], x[1][0][2], np.dot(x[1][0][1], x[1][1]), np.dot(x[1][0][1], x[1][1]) - x[1][0][2]) ))

  # Then we get a key value with keys = (i,j), and values = (u,v) i.e. the index of user/product and the associated vectors
  #u_v_vect = u_v.map(lambda x : ( (x[0][0],x[1][0]) , (x[0][1],x[1][1]) ) ) 
  #u_v_vect.persist()

  #u_v_dot =  u_v.map(lambda x : ( (x[0][0],x[1][0]) , np.dot(x[0][1],x[1][1]) ) )
  
  # We then take the difference between the prediction and the real value
  #diff = u_v_y_dot.mapValues(lambda x : (x[3] - x[2]))
  #diff.persist()

  # We compute the mean squared loss
  loss = u_v_y_dot.mapValues(lambda x : x[4]**2).map(lambda x: x[1]).mean() / 2
  
  # We print the loss
  print("Current loss at epoch %i : %f" % (l,loss))
  
  # We compute the gradients for each pair
  #grads = u_v_vect.join(diff)
  #grads.persist()

  # We then compute the gradients wrt u and v
  grad_u = u_v_y_dot.map(lambda x : (x[0][0], x[1][4] * x[1][1]))
  grad_v = u_v_y_dot.map(lambda x : (x[0][1], x[1][4] * x[1][0]))
  
  # We take the average of each batch
  agg_grad_u = grad_u.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  agg_grad_v = grad_v.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  
  # We make one step of a gradient descent
  u_rows = u_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1]).collect()
  v_rows = v_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1]).collect()
  
  # Free u_v_vect and diff
  #u_v_vect.unpersist()
  #diff.unpersist()
  #grads.unpersist()

# COMMAND ----------

"""Version 3 : Coalesce"""

import numpy as np

# HHYPER-PARAMETERS
N_WORKERS = 8
N_EPOCH = 25
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

# We create an RDD with key = (i,j) and values = rating
y = df.rdd.map(lambda x : ( (x["userId"],x["movieId"]), int(x["rating"])) )

# We then broadcast it
sc.broadcast(y.collect())

for l in range(N_EPOCH):
  
  # First we map the observations with the local U matrix
  a = y.map(lambda x : (x[0][0], (x[0][1] , x[1]) )).join(u_rows)
  
  # Then we reorder in order to join with the V matrix
  b = a.map(lambda x : (x[1][0][0], (x[0], x[1][1], x[1][0][1]) ))
  
  # We then join and reorder into ((i,j) ,(u,v,y,y_pred,diff))
  c = b.join(v_rows)
  u_v_y_dot = c.map(lambda x : ((x[1][0][0], x[0]) , (x[1][0][1], x[1][1], x[1][0][2], np.dot(x[1][0][1], x[1][1]), np.dot(x[1][0][1], x[1][1]) - x[1][0][2]) ))

  # Then we get a key value with keys = (i,j), and values = (u,v) i.e. the index of user/product and the associated vectors
  #u_v_vect = u_v.map(lambda x : ( (x[0][0],x[1][0]) , (x[0][1],x[1][1]) ) ) 
  #u_v_vect.persist()

  #u_v_dot =  u_v.map(lambda x : ( (x[0][0],x[1][0]) , np.dot(x[0][1],x[1][1]) ) )
  
  # We then take the difference between the prediction and the real value
  #diff = u_v_y_dot.mapValues(lambda x : (x[3] - x[2]))
  #diff.persist()

  # We compute the mean squared loss
  loss = u_v_y_dot.mapValues(lambda x : x[4]**2).map(lambda x: x[1]).mean() / 2
  
  # We print the loss
  print("Current loss at epoch %i : %f" % (l,loss))
  
  # We compute the gradients for each pair
  #grads = u_v_vect.join(diff)
  #grads.persist()

  # We then compute the gradients wrt u and v
  grad_u = u_v_y_dot.map(lambda x : (x[0][0], x[1][4] * x[1][1]))
  grad_v = u_v_y_dot.map(lambda x : (x[0][1], x[1][4] * x[1][0]))
  
  # We take the average of each batch
  agg_grad_u = grad_u.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  agg_grad_v = grad_v.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  
  # We make one step of a gradient descent
  u_rows = u_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1])
  v_rows = v_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1])
  
  u_rows.coalesce(N_WORKERS)
  v_rows.coalesce(N_WORKERS)
  
  # Free u_v_vect and diff
  #u_v_vect.unpersist()
  #diff.unpersist()
  #grads.unpersist()

# COMMAND ----------

y = df.rdd.map(lambda x : ( (x["userId"],x["movieId"]), int(x["rating"])) )
keys = y.keys()
keys = keys.map(lambda x : (x[0],1))
keys = keys.reduceByKey(lambda x,y : x + y)

df_test = spark.createDataFrame(keys).toDF("userId", "count")
df_test.createTempView("test3")

# Create a RDD with cumulative count of total movies
usr_to_count = sqlContext.sql("select userId,count, " +
  "SUM(count) over (  order by userId  rows between unbounded preceding and current row ) cum_count " +
  "from test3")

usr_cum_count = usr_to_count.rdd.map(lambda x : x["cum_count"]).collect()

def count_partition(key):
  key = int(( float(usr_cum_count[key]) / N) * N_WORKERS)
  return(key)

# COMMAND ----------

"""Version 3 : partition by key"""

import numpy as np

# HHYPER-PARAMETERS
N_WORKERS = 8
N_EPOCH = 25
step = 0.01
hidden_size = 5

n = len(usr_to_emb)
m = len(movie_to_emb)

# Matrixes which represent our embeddings
U = np.random.randn(n,hidden_size)
V = np.random.randn(m,hidden_size)
N = df.count()

# We then create our RDDs from the matrixes
u_rows = sc.parallelize((i,U[i]) for i in range(n))
v_rows = sc.parallelize((i,V[i]) for i in range(m))

# We create an RDD with key = (i,j) and values = rating
y = df.rdd.map(lambda x : ( (x["userId"],x["movieId"]), int(x["rating"])) )

# We then broadcast it
sc.broadcast(y.collect())

for l in range(N_EPOCH):
  
  if l > 0:
    u_rows = sc.parallelize(u_rows)
    v_rows = sc.parallelize(v_rows)
  
  u_rows.partitionBy(N_WORKERS,count_partition)
  
  # First we map the observations with the local U matrix
  a = y.map(lambda x : (x[0][0], (x[0][1] , x[1]) )).join(u_rows)
  
  # Then we reorder in order to join with the V matrix
  b = a.map(lambda x : (x[1][0][0], (x[0], x[1][1], x[1][0][1]) ))
  
  # We then join and reorder into ((i,j) ,(u,v,y,y_pred,diff))
  c = b.join(v_rows)
  u_v_y_dot = c.map(lambda x : ((x[1][0][0], x[0]) , (x[1][0][1], x[1][1], x[1][0][2], np.dot(x[1][0][1], x[1][1]), np.dot(x[1][0][1], x[1][1]) - x[1][0][2]) ))

  # Then we get a key value with keys = (i,j), and values = (u,v) i.e. the index of user/product and the associated vectors
  #u_v_vect = u_v.map(lambda x : ( (x[0][0],x[1][0]) , (x[0][1],x[1][1]) ) ) 
  #u_v_vect.persist()

  #u_v_dot =  u_v.map(lambda x : ( (x[0][0],x[1][0]) , np.dot(x[0][1],x[1][1]) ) )
  
  # We then take the difference between the prediction and the real value
  #diff = u_v_y_dot.mapValues(lambda x : (x[3] - x[2]))
  #diff.persist()

  # We compute the mean squared loss
  loss = u_v_y_dot.mapValues(lambda x : x[4]**2).map(lambda x: x[1]).mean() / 2
  
  # We print the loss
  print("Current loss at epoch %i : %f" % (l,loss))
  
  # We compute the gradients for each pair
  #grads = u_v_vect.join(diff)
  #grads.persist()

  # We then compute the gradients wrt u and v
  grad_u = u_v_y_dot.map(lambda x : (x[0][0], x[1][4] * x[1][1]))
  grad_v = u_v_y_dot.map(lambda x : (x[0][1], x[1][4] * x[1][0]))
  
  # We take the average of each batch
  agg_grad_u = grad_u.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  agg_grad_v = grad_v.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  
  # We make one step of a gradient descent
  u_rows = u_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1]).collect()
  v_rows = v_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1]).collect()
  
  # Free u_v_vect and diff
  #u_v_vect.unpersist()
  #diff.unpersist()
  #grads.unpersist()

# COMMAND ----------

"""Version 2 : deleted cartesian product and optimized join"""

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

# We create an RDD with key = (i,j) and values = rating
y = df.rdd.map(lambda x : ( (x["userId"],x["movieId"]), int(x["rating"])) )

# We then broadcast it
sc.broadcast(y.collect())

for l in range(N_EPOCH):
  
  if l > 0:
    u_rows = sc.parallelize(u_rows)
    v_rows = sc.parallelize(v_rows)
  
  # First we map the observations with the local U matrix
  a = y.map(lambda x : (x[0][0], (x[0][1] , x[1]) )).join(u_rows)
  
  # Then we reorder in order to join with the V matrix
  b = a.map(lambda x : (x[1][0][0], (x[0], x[1][1], x[1][0][1]) ))
  
  # We then join and reorder into ((i,j) ,(u,v,y,y_pred,diff))
  c = b.join(v_rows)
  u_v_y_dot = c.map(lambda x : ((x[1][0][0], x[0]) , (x[1][0][1], x[1][1], x[1][0][2], 1 + 4 * sigmoid(np.dot(x[1][0][1], x[1][1])), np.dot(x[1][0][1], x[1][1]) - x[1][0][2],sigmoid(np.dot(x[1][0][1], x[1][1])) ) ) )

  # Then we get a key value with keys = (i,j), and values = (u,v) i.e. the index of user/product and the associated vectors
  #u_v_vect = u_v.map(lambda x : ( (x[0][0],x[1][0]) , (x[0][1],x[1][1]) ) ) 
  #u_v_vect.persist()

  #u_v_dot =  u_v.map(lambda x : ( (x[0][0],x[1][0]) , np.dot(x[0][1],x[1][1]) ) )
  
  # We then take the difference between the prediction and the real value
  #diff = u_v_y_dot.mapValues(lambda x : (x[3] - x[2]))
  #diff.persist()

  # We compute the mean squared loss
  loss = u_v_y_dot.mapValues(lambda x : x[4]**2).map(lambda x: x[1]).mean() / 2
  
  # We print the loss
  print("Current loss at epoch %i : %f" % (l,loss))
  
  # We compute the gradients for each pair
  #grads = u_v_vect.join(diff)
  #grads.persist()

  # We then compute the gradients wrt u and v
  grad_u = u_v_y_dot.map(lambda x : (x[0][0], 4 * x[1][4] * x[1][5] * (1 - x[1][5]) * x[1][1]))
  grad_v = u_v_y_dot.map(lambda x : (x[0][1], 4 * x[1][4] * x[1][5] * (1 - x[1][5]) * x[1][0]))
  
  # We take the average of each batch
  agg_grad_u = grad_u.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  agg_grad_v = grad_v.map(lambda x : (x[0], (1,x[1]))).reduceByKey(lambda x,y : (x[0] + y[0],x[1] + y[1])).mapValues(lambda x : x[1] / x[0])
  
  # We make one step of a gradient descent
  u_rows = u_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1]).collect()
  v_rows = v_rows.join(agg_grad_u).mapValues(lambda x : x[0] - step * x[1]).collect()
  
  # Free u_v_vect and diff
  #u_v_vect.unpersist()
  #diff.unpersist()
  #grads.unpersist()

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

y.count()

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