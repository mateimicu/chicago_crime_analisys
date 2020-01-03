import pandas
import random

filename = "Crimes_-_2001_to_present.csv"
n = sum(1 for line in open(filename)) - 1
s = 300000
skip = sorted(random.sample(range(1,n+1),n-s))
df = pandas.read_csv(filename, skiprows=skip)
df.to_csv('sampled_data.csv')