from pyspark import SparkConf, SparkContext
import numpy as np

np1 = np.array([1,2,3])

conf = SparkConf().setAppName('My App')
sc = SparkContext(conf=conf)

count = sc.range(1, 1000 * 1000 * 100).filter(lambda x: x > 100).count()
print('count: ', count)
