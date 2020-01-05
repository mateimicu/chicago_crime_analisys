from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("Cluster Crimes").getOrCreate()
df = spark.read.csv('sampled_data.csv', inferSchema=True, header=True)
df = df.cache()
aux = df.columns
for it in aux:
    df = df.withColumnRenamed(it, it.lower().replace(' ', '_'))
df = df.where(df.latitude.isNotNull() & df.longitude.isNotNull() & df.location_description.isNotNull()).cache()
df = df.where(df['x_coordinate'] != 0.0)
df = df.where(df['y_coordinate'] != 0.0)
df = df.withColumn('arrest', df['arrest'].cast("string"))
df = df.withColumn('domestic', df['domestic'].cast("string"))
df = df.cache()

df = df.withColumn('date_time', to_timestamp('date', 'MM/dd/yyyy hh:mm:ss a'))\
       .withColumn('month', trunc('date_time', 'YYYY'))
df_hour = df.withColumn('hour', hour(df['date_time']))
df_dates = df_hour.withColumn('week_day', dayofweek(df_hour['date_time']))\
                 .withColumn('year_month', month(df_hour['date_time']))\
                 .withColumn('month_day', dayofmonth(df_hour['date_time']))\
                 .withColumn('date_number', datediff(df['date_time'], to_date(lit('2001-01-01'), format='yyyy-MM-dd')))\
                 .cache()

selected_features = [
 'location_description',
 'arrest',
 'domestic',
 'beat',
 'district',
 'ward',
 'community_area',
 'fbi_code',
 'hour',
 'week_day',
 'year_month',
 'month_day',
 'date_number']

features_df = df_dates.select(selected_features)
feature_level_count_dic = []
for feature in selected_features:
    levels_list_df = features_df.select(feature).distinct()
    feature_level_count_dic.append({'feature': feature, 'level_count': levels_list_df.count()})
    
df_dates_features = df_dates.na.drop(subset=selected_features)
for feature in feature_level_count_dic:
    indexer = StringIndexer(inputCol=feature['feature'], outputCol='%s_indexed' % feature['feature'])
    model = indexer.fit(df_dates_features)
    df_dates_features = model.transform(df_dates_features)

response_indexer = StringIndexer(inputCol='primary_type', outputCol='primary_type_indexed')
response_model = response_indexer.fit(df_dates_features)
df_dates_features = response_model.transform(df_dates_features)
indexed_features = ['%s_indexed' % fc['feature'] for fc in feature_level_count_dic]
assembler = VectorAssembler(inputCols=indexed_features, outputCol='features')
vectorized_df_dates = assembler.transform(df_dates_features)
train, test = vectorized_df_dates.randomSplit([0.8, 0.2])
logisticRegression = LogisticRegression(labelCol='primary_type_indexed', featuresCol='features', maxIter=50, family='multinomial')
fittedModel = logisticRegression.fit(train)
evaluation_summary = fittedModel.evaluate(test)
print(evaluation_summary.accuracy)