from K_means import *
from Visualisation import *
from pandas import read_csv
import numpy as np

data = read_csv("midtermProject-part2-data.csv")
data = NormalizeData(data)


Centroids,Point_Clusters = CalculateKmeans(data,k=3,epoch=1000)

data_cluster_visualization(0,3,Point_Clusters,data)
