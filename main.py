from sqlite3 import OperationalError

import psycopg2
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as ps
from pyspark.sql.types import StructType, StructField, NumericType, IntegerType, StringType, TimestampType, FloatType
from sqlalchemy import create_engine
from pyspark.sql.window import Window

appName = "task"
master = "local"

spark = SparkSession.builder.master(master).appName(appName).getOrCreate()
engine = create_engine(
    "postgresql+psycopg2://postgres:secret@localhost/pag?client_encoding=utf8")
category = spark.createDataFrame(pd.read_sql('select * from category', engine))
film_category = spark.createDataFrame(pd.read_sql('select * from film_category', engine))
film_actor = spark.createDataFrame(pd.read_sql('select * from film_actor', engine))
actor = spark.createDataFrame(pd.read_sql('select * from actor', engine))
schema = StructType([StructField("film_id",IntegerType(), True), StructField("title", StringType(), True), StructField("description", StringType(), True), StructField("release_year", IntegerType(), True), StructField("language_id", IntegerType(), True), StructField("original_language_id", IntegerType(), True), StructField("rental_duration", IntegerType(), True), StructField("rental_rate", FloatType(), True), StructField("length", IntegerType(), True), StructField("replacement_cost", FloatType(), True), StructField("rating", StringType(), True), StructField("last_update", TimestampType(), True), StructField("special_features", StringType(), True), StructField("fulltext", StringType(), True)])
film = spark.createDataFrame(pd.read_sql('select * from film', engine), schema=schema)
rental = spark.createDataFrame(pd.read_sql('select * from rental', engine))
inventory = spark.createDataFrame(pd.read_sql('select * from inventory', engine))
payment = spark.createDataFrame(pd.read_sql('select * from payment', engine))
address = spark.createDataFrame(pd.read_sql('select * from address', engine))
city = spark.createDataFrame(pd.read_sql('select * from city', engine))
customer = spark.createDataFrame(pd.read_sql('select * from customer', engine))

# 1
data = category.join(film_category, category.category_id == film_category.category_id, "inner")
data.groupBy("name").agg(ps.count("*").alias("films")).orderBy('films', ascending=False).show()

# 2
data = film.join(film_actor, film.film_id == film_actor.film_id, "inner")\
    .join(actor, actor.actor_id == film_actor.actor_id, "inner" )
data.groupBy("first_name", "last_name").agg(ps.count("rental_duration")
                                            .alias("rentals")).orderBy('rentals', ascending=False).show(10)

# 3
data = payment.join(rental, payment.rental_id == rental.rental_id)\
    .join(inventory, inventory.inventory_id == rental.inventory_id)\
    .join(film_category, inventory.film_id == film_category.film_id)\
    .join(category, film_category.category_id == category.category_id)
data.groupBy("name").agg(ps.sum("amount").alias("s")).orderBy("s", ascending=False).show(1)

# 4
data = film.join(inventory, film.film_id == inventory.film_id, "left").where(inventory.film_id.isNull())
data.orderBy("title").select("title").show()

# 5
data = category.join(film_category, category.category_id == film_category.category_id, "inner")\
    .join(film, film_category.film_id == film.film_id)\
    .join(film_actor, film_actor.film_id == film.film_id)\
    .join(actor, actor.actor_id == film_actor.actor_id)\
    .where(category.name == "Children")
data = data.groupBy("first_name", "last_name").agg(ps.count("name")
                                                   .alias("top")).orderBy("top", ascending=False)
w = Window.orderBy(data.top.desc())
data = data.withColumn("newtop", ps.dense_rank().over(w))
data.where(data.newtop <= 3).show()

# 6
data = city.join(address, city.city_id == address.city_id)\
    .join(customer, address.address_id == customer.address_id)
data.groupBy("city")\
    .agg(
    ps.sum(ps.when(data.active != 1, 1).otherwise(0)).alias("inactive"),
    ps.sum(ps.when(data.active == 1, 1).otherwise(0)).alias("active"))\
    .orderBy("active", ascending=True).show()

# 7
data = rental.join(inventory, rental.inventory_id == inventory.inventory_id)\
    .join(film_category, inventory.film_id == film_category.film_id)\
    .join(category, film_category.category_id == category.category_id)\
    .join(customer, customer.customer_id == rental.customer_id)\
    .join(address, customer.address_id == address.address_id)\
    .join(city, address.city_id == city.city_id)
data = data.groupBy("city", "name")\
    .agg(ps.sum(((ps.unix_timestamp(data.return_date) - ps.unix_timestamp(data.rental_date))/3600))
         .alias("hours")).orderBy("hours", ascending=True)
data1 = data.selectExpr("city as c", "hours as h", "name as n")
data = data.join(data1, ps.when(data.city == data1.c, True) & ps.when(data.hours < data1.h, True), "left")\
    .where(data1.h.isNull() & data.hours.isNotNull())
data.select("city", "name", "hours").orderBy("city").where(data.city.like('A%') | data.city.like('%-%')).show()