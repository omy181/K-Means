from K_means import *
from pandas import read_csv

data = read_csv("midtermProject-part2-data.csv")

CalculateKmeans(data,k=3,epoch=1)