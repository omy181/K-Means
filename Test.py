from K_means import *
from pandas import read_csv
import numpy as np

data = read_csv("midtermProject-part2-data.csv")
data = NormalizeData(data)


Centroids = CalculateKmeans(data,k=3,epoch=2)



#AskQuestion(Centroids,data.to_numpy()[0])