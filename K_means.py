import pandas as pd
import numpy as np
import math as math

def FindClosestCentroid(centroids,point):
    closest_distance = 0
    closest_centroid = None

    for centroid in centroids:
        #calculate distances
        varsum = 0
        for c_var,p_var in zip(centroid,point):
            varsum += pow(c_var-p_var,2)

        distance = math.sqrt(varsum)

        if(closest_distance == 0 or distance < closest_distance):
            closest_distance = distance
            closest_centroid = centroid

    return closest_centroid

def CalculateKmeans(data:pd.DataFrame,k,epoch):
    centroids = data.sample(k).to_numpy()   
    points = data.to_numpy()

    Point_Clusters = {}

    for point in points:
        Closest_Centroid = FindClosestCentroid(centroids,point)
        Point_Clusters[point] = Closest_Centroid #dictionary index is not good find a better way to put index to the points

    print(Point_Clusters)

    return 

