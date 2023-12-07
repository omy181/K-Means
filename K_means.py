import pandas as pd
import numpy as np

def CalculateKmeans(data:pd.DataFrame,k,epoch):
    centroids = data.sample(k).to_numpy()
    
    FindClosestCentroid(centroids,centroids[0])
    return

def FindClosestCentroid(centroids,point):
    for centroid in centroids:
        #calculate distances
        print(centroid)
    return