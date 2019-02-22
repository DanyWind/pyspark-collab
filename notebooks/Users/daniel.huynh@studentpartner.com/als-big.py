# Databricks notebook source
# MAGIC %md ## Database management project

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

# First we compute the unique users

unique_usr = df.select('userId').distinct().collect()
unique_usr = [row.asDict()["userId"] for row in unique_usr]

#usr_to_emb = [(usr, i) for i,usr in enumerate(unique_usr)]
#emb_to_usr = [(i,usr) for i,usr in enumerate(unique_usr)]

usr_to_emb = {usr : i for i,usr in enumerate(unique_usr)}
emb_to_usr = {i : usr for i,usr in enumerate(unique_usr)}
unique_movie = df.select('movieId').distinct().collect()
unique_movie = [int(row.asDict()["movieId"]) for row in unique_movie]

#movie_to_emb = [(movie, i) for i,movie in enumerate(unique_movie)]
#emb_to_movie = [(i, movie) for i,movie in enumerate(unique_movie)]

movie_to_emb = {movie : i for i,movie in enumerate(unique_movie)}
emb_to_movie = {i : movie for i,movie in enumerate(unique_movie)}

import numpy as np

# HHYPER-PARAMETERS
N_WORKERS = 8
N_EPOCH = 100
step = 0.01
hidden_size = 5

n = len(usr_to_emb)
m = len(movie_to_emb)

# Matrixes which represent our embeddings
U = np.random.randn(n,hidden_size)
V = np.random.randn(m,hidden_size)

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

"""Version 2 : deleted cartesian product and optimized join"""

import time

times2 = []
start = time.time()

losses2 = []

# We then create our RDDs from the matrixes
u_rows = sc.parallelize((i,U[i]) for i in range(n))
v_rows = sc.parallelize((i,V[i]) for i in range(m))

# We create an RDD with key = (i,j) and values = rating
y = df.rdd.map(lambda x : (usr_to_emb[x["userId"]], movie_to_emb[x["movieId"]], x["rating"]))

# We then broadcast it
# sc.broadcast(y.collect())

for l in range(N_EPOCH):
  
  if l > 0:
    u_rows = sc.parallelize(u_rows)
    v_rows = sc.parallelize(v_rows)
  
  # First we map the observations with the local U matrix
  a = y.map(lambda x : (x[0], (x[1],x[2]))).join(u_rows)
  
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
  
  diff = time.time() - start
  times2.append(diff)
  
  losses2.append(loss)
  
  #u_rows.coalesce(N_WORKERS)
  #v_rows.coalesce(N_WORKERS)
  
  # Free u_v_vect and diff
  #u_v_vect.unpersist()
  #diff.unpersist()
  #grads.unpersist()

# COMMAND ----------

# MAGIC %md Dataset : small<br>
# MAGIC Time ; 5.27  minutes<br>
# MAGIC Epochs : 100<br>
# MAGIC Loss : 3.5

# COMMAND ----------

# MAGIC %md Dataset : big<br>
# MAGIC Time ; 8.59 m<br>
# MAGIC Epochs : 3<br>
# MAGIC Loss : 9.12

# COMMAND ----------

# MAGIC %md Dataset : big <br>
# MAGIC Time : 14.26 m<br>
# MAGIC Epochs : 5<br>
# MAGIC Loss : 9.05

# COMMAND ----------

"""Version 3 : added coalesce"""

import time

times3 = []
start = time.time()

# We then create our RDDs from the matrixes
u_rows = sc.parallelize((i,U[i]) for i in range(n))
v_rows = sc.parallelize((i,V[i]) for i in range(m))

# We create an RDD with key = (i,j) and values = rating
y = df.rdd.map(lambda x : (usr_to_emb[x["userId"]], movie_to_emb[x["movieId"]], x["rating"]))

# We then broadcast it
# sc.broadcast(y.collect())

for l in range(N_EPOCH):
  
  # First we map the observations with the local U matrix
  a = y.map(lambda x : (x[0], (x[1],x[2]))).join(u_rows)
  
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
  
  diff = time.time() - start
  times3.append(diff)
  
  # Free u_v_vect and diff
  #u_v_vect.unpersist()
  #diff.unpersist()
  #grads.unpersist()

# COMMAND ----------

# MAGIC %md Dataset : big<br>
# MAGIC Time : 15.83<br>
# MAGIC Epochs : 5<br>
# MAGIC Loss : 9.04

# COMMAND ----------

# MAGIC %md Dataset : big<br>
# MAGIC Time : 7.57<br>
# MAGIC Epochs : 3<br>
# MAGIC Loss : 9.19

# COMMAND ----------

"""Version 4 : added broadcasting"""

import time

times4= []
start = time.time()
losses4 = []

# We then create our RDDs from the matrixes
u_rows = sc.parallelize((i,U[i]) for i in range(n))
v_rows = sc.parallelize((i,V[i]) for i in range(m))

# First we do the mapping between the raw indexes to the normalized indexes
y = df.rdd.map(lambda x : (usr_to_emb[x["userId"]], (movie_to_emb[x["movieId"]], x["rating"])))

# Then we group by the key i of users and broadcast the dictionary (i, [(j,yij), ... ])
y_broadcast = y.groupByKey().collectAsMap()
y_broadcast = sc.broadcast(y_broadcast)

for l in range(N_EPOCH):
  
  if l > 0:
    u_rows = sc.parallelize(u_rows)
    v_rows = sc.parallelize(v_rows)
  
  # First we do a lookup with the broadcast y  
  a = u_rows.map(lambda x : ((x[0],x[1]), y_broadcast.value[x[0]]))
  
  # We then flatMap values of the elements x : ((i,u), [(j,y)]) and then we reorder it to join on the j 
  b = a.flatMapValues(lambda x : x).map(lambda x : ( x[1][0], (x[0][0], x[0][1], x[1][1])))
  
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
  
  diff = time.time() - start
  times4.append(diff)
  
  losses4.append(loss)
  
  #u_rows.coalesce(N_WORKERS)
  #v_rows.coalesce(N_WORKERS)
  
  # Free u_v_vect and diff
  #u_v_vect.unpersist()
  #diff.unpersist()
  #grads.unpersist()

# COMMAND ----------

# MAGIC %md Dataset : small<br>
# MAGIC Epoch : 100<br>
# MAGIC Time : 2.64<br>
# MAGIC Loss : 3.6

# COMMAND ----------

# MAGIC %md
# MAGIC Dataset : small 
# MAGIC Epoch : 50
# MAGIC Time : 1.33 minutes
# MAGIC Loss : 5.6

# COMMAND ----------

"""Version 5 : added partitioning"""
import time

times5 = []
start = time.time()
losses5 = []

# We then create our RDDs from the matrixes
u_rows = sc.parallelize((i,U[i]) for i in range(n))
v_rows = sc.parallelize((i,V[i]) for i in range(m))

# First we do the mapping between the raw indexes to the normalized indexes
y = df.rdd.map(lambda x : (usr_to_emb[x["userId"]], (movie_to_emb[x["movieId"]], x["rating"])))

# Then we group by the key i of users and broadcast the dictionary (i, [(j,yij), ... ])
y_broadcast = y.groupByKey().collectAsMap()
y_broadcast = sc.broadcast(y_broadcast)

y = df.rdd.map(lambda x : ( (x["userId"],x["movieId"]), int(x["rating"])) )
keys = y.keys()
keys = keys.map(lambda x : (x[0],1))
keys = keys.reduceByKey(lambda x,y : x + y)

df_test = spark.createDataFrame(keys).toDF("userId", "count")
df_test.createTempView("test")

# Create a RDD with cumulative count of total movies
usr_to_count = sqlContext.sql("select userId,count, " +
  "SUM(count) over (  order by userId  rows between unbounded preceding and current row ) cum_count " +
  "from test")

usr_cum_count = usr_to_count.rdd.map(lambda x : x["cum_count"]).collect()

N = df.count()

def count_partition(key):
  key = int(( float(usr_cum_count[key]) / N) * N_WORKERS)
  return(key)

for l in range(N_EPOCH):
  
  if l > 0:
    u_rows = sc.parallelize(u_rows)
    v_rows = sc.parallelize(v_rows)
    
  # Partitioning
  u_rows = u_rows.partitionBy(N_WORKERS,count_partition)
  
  # First we do a lookup with the broadcast y  
  a = u_rows.map(lambda x : ((x[0],x[1]), y_broadcast.value[x[0]]))
  
  # We then flatMap values of the elements x : ((i,u), [(j,y)]) and then we reorder it to join on the j 
  b = a.flatMapValues(lambda x : x).map(lambda x : ( x[1][0], (x[0][0], x[0][1], x[1][1])))
  
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
  
  diff = time.time() - start
  times5.append(diff)
  
  losses5.append(loss)
  
  #u_rows.coalesce(N_WORKERS)
  #v_rows.coalesce(N_WORKERS)
  
  # Free u_v_vect and diff
  #u_v_vect.unpersist()
  #diff.unpersist()
  #grads.unpersist()

# COMMAND ----------

# MAGIC %md Dataset : small<br>
# MAGIC Time : 2.89<br>
# MAGIC Epochs : 100<br>
# MAGIC Loss : 3.6

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
line1, = ax.plot(times2,label= "Vanilla")
line2, = ax.plot(times4,label = "Broadcast")
line3, = ax.plot(times5, label = "Broadcast + partition")

ax.set_xlabel("Epochs")
ax.set_ylabel("Time")
ax.set_title("Performances on the small dataset")

ax.legend(handles = [line1,line2,line3],loc = 1)

display(fig)

# COMMAND ----------

fig, ax = plt.subplots()
line1, = ax.plot(losses2,label= "Vanilla")
line2, = ax.plot(losses4,label = "Broadcast")
line3, = ax.plot(losses5, label = "Broadcast + partition")

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Performances on the small dataset")

ax.legend(handles = [line1,line2,line3],loc = 1)

display(fig)