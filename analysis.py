from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import *
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


spark = SparkSession.builder.appName("Cluster Crimes").getOrCreate()
df = spark.read.csv(sys.argv[1], inferSchema=True, header=True)
df = df.cache()
aux = df.columns
for it in aux:
    df = df.withColumnRenamed(it, it.lower().replace(' ', '_'))
df = df.where(df.latitude.isNotNull() & df.longitude.isNotNull() & df.location_description.isNotNull()).cache()
df = df.where(df['x_coordinate'] != 0.0)
df = df.where(df['y_coordinate'] != 0.0)
df = df.cache()

plt.clf()
crime_type_groups = df.groupBy('primary_type').count()
crime_type_counts = crime_type_groups.orderBy('count', ascending=False)
counts_pddf = pd.DataFrame(crime_type_counts.rdd.map(lambda l: l.asDict()).collect())
plt.rcParams["figure.figsize"] = [20, 8]
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
type_graph = sns.barplot(x='count', y='primary_type', data=counts_pddf)
type_graph.set(ylabel="Primary Type", xlabel="Crimes Record Count")
type_graph.get_figure().savefig('crimes_histogram.pdf')

df = df.withColumn('date_time', to_timestamp('date', 'MM/dd/yyyy hh:mm:ss a'))\
       .withColumn('month', trunc('date_time', 'YYYY'))
type_arrest_date = df.groupBy(['arrest', 'month'])\
                     .count()\
                     .orderBy(['month', 'count'], ascending=[True, False])
type_arrest_pddf = pd.DataFrame(type_arrest_date.rdd.map(lambda l: l.asDict()).collect())
type_arrest_pddf['yearpd'] = type_arrest_pddf['month'].apply(lambda dt: datetime.datetime.strftime(pd.Timestamp(dt), '%Y'))
type_arrest_pddf['arrest'] = type_arrest_pddf['arrest'].apply(lambda l: l=='True')

plt.clf()
arrested = type_arrest_pddf[type_arrest_pddf['arrest'] == True]
not_arrested = type_arrest_pddf[type_arrest_pddf['arrest'] == False]
fig, ax = plt.subplots()
ax.plot(arrested['month'], arrested['count'], label='Arrested')
ax.plot(not_arrested['month'], not_arrested['count'], label='Not Arrested')
ax.set(xlabel='Year - 2001-2019', ylabel='Total records', title='Year-on-year crime records')
ax.grid(b=True, which='both', axis='y')
fig.savefig("arrest_evolution.pdf")

plt.clf()
df_hour = df.withColumn('hour', hour(df['date_time']))
hourly_count = df_hour.groupBy(['primary_type', 'hour']).count().cache()
hourly_total_count = hourly_count.groupBy('hour').sum('count')
hourly_count_pddf = pd.DataFrame(hourly_total_count.select(hourly_total_count['hour'], hourly_total_count['sum(count)'].alias('count'))\
                                .rdd.map(lambda l: l.asDict()).collect())
hourly_count_pddf = hourly_count_pddf.sort_values(by='hour')
fig, ax = plt.subplots()
ax.plot(hourly_count_pddf['hour'], hourly_count_pddf['count'], label='Hourly Count')
ax.set(xlabel='Hour of Day', ylabel='Total records',title='Overall hourly crime numbers')
ax.grid(b=True, which='both', axis='y')
fig.savefig("best_crime_hours.pdf")

plt.clf()
location_type_groups = df.groupBy(['location_description']).count()
location_type_counts = location_type_groups.orderBy('count', ascending=False)
loc_counts_pddf = pd.DataFrame(location_type_counts.rdd.map(lambda l: l.asDict()).collect())
plt.rcParams["figure.figsize"] = [20, 8]
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
loc_type_graph = sns.barplot(x='count', y='location_description', data=loc_counts_pddf)
loc_type_graph.set(ylabel="Location", xlabel="Crimes Record Count")
loc_type_graph.get_figure().savefig('location_histogram.pdf')

plt.clf()
location_hour = df_hour.groupBy(['location_description', 'hour']).count().orderBy('count', ascending=False)
street_home_hour = location_hour.where((location_hour['location_description'] == 'STREET') | (location_hour['location_description'] == 'RESIDENCE'))
street_home_hour_pddf = pd.DataFrame(street_home_hour.rdd.map(lambda row: row.asDict()).collect())
street_home_hour_pddf = street_home_hour_pddf.sort_values(by='hour')
fig, ax = plt.subplots()
ax.plot(street_home_hour_pddf[street_home_hour_pddf['location_description'] == 'RESIDENCE']['hour'], 
          street_home_hour_pddf[street_home_hour_pddf['location_description'] == 'RESIDENCE']['count'],
         label='Crimes at home')
ax.plot(street_home_hour_pddf[street_home_hour_pddf['location_description'] == 'STREET']['hour'], 
          street_home_hour_pddf[street_home_hour_pddf['location_description'] == 'STREET']['count'],
         label='Crimes on the street')
ax.grid(b=True, which='both', axis='y')
fig.savefig("home_vs_street_crimes.pdf")

plt.clf()
df_dates = df_hour.withColumn('week_day', dayofweek(df_hour['date_time']))\
                 .withColumn('year_month', month(df_hour['date_time']))\
                 .withColumn('month_day', dayofmonth(df_hour['date_time']))\
                 .withColumn('date_number', datediff(df['date_time'], to_date(lit('2001-01-01'), format='yyyy-MM-dd')))\
                 .cache()
week_day_crime_counts = df_dates.groupBy('week_day').count()
week_day_crime_counts_pddf = pd.DataFrame(week_day_crime_counts.orderBy('week_day').rdd.map(lambda e: e.asDict()).collect())
sns.barplot(data=week_day_crime_counts_pddf, x='week_day', y='count').get_figure().savefig('week_crime_histo.pdf')

plt.clf()
year_month_crime_counts = df_dates.groupBy('year_month').count()
year_month_crime_counts_pddf = pd.DataFrame(year_month_crime_counts.orderBy('year_month').rdd.map(lambda e: e.asDict()).collect())
sns.barplot(data=year_month_crime_counts_pddf, y='count', x='year_month').get_figure().savefig('month_crime_histo.pdf')

plt.clf()
month_day_crime_counts = df_dates.groupBy('month_day').count()
month_day_crime_counts_pddf = pd.DataFrame(month_day_crime_counts.orderBy('month_day').rdd.map(lambda e: e.asDict()).collect())
month_day_crime_counts_pddf = month_day_crime_counts_pddf.sort_values(by='month_day', ascending=True)
fg, ax = plt.subplots()
ax.plot(month_day_crime_counts_pddf['month_day'], month_day_crime_counts_pddf['count'], label='Crimes over the month')
ax.grid(b=True, which='both')
ax.legend()
fg.savefig('best_day_crime.pdf')

