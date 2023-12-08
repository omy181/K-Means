import pandas as pd
import numpy as np
import math as math
import pprint as pp

def FindClosestCentroid(centroids,point):
    closest_distance = 0
    closest_centroid_index = 0

    for i,centroid in enumerate(centroids):
        #calculate distances
        varsum = 0
        for c_var,p_var in zip(centroid,point):
            varsum += pow(c_var-p_var,2)

        distance = math.sqrt(varsum)

        if(closest_distance == 0 or distance < closest_distance):
            closest_distance = distance
            closest_centroid_index = i

    return closest_centroid_index

def AssigntoNearestCentroid(centroids,points):
    Point_Clusters = {}
    for i in range(len(centroids)): 
        Point_Clusters[i] = []

    for point in points:
        Closest_Centroid_index = FindClosestCentroid(centroids,point)
        Point_Clusters[Closest_Centroid_index].append(point)
    
    return Point_Clusters

def FindNewCentroids(Point_Clusters):
    centroids = []
    for Cluster_Index,Cluster in zip(Point_Clusters,Point_Clusters.values()):
        Sum = 0
        for point in Cluster:
            Sum += point
        Sum /= len(Cluster)
        centroids.append(Sum)


    return centroids

def CalculateKmeans(data:pd.DataFrame,k,epoch):
    centroids = data.sample(k).to_numpy()   
    points = data.to_numpy()

    Point_Clusters = None


    for e in range(epoch):
        Point_Clusters = AssigntoNearestCentroid(centroids,points)
        centroids = FindNewCentroids(Point_Clusters)

    return centroids

def AskQuestion(Centroids,Question_Point):
    i = FindClosestCentroid(Centroids,Question_Point)
    print("\n\nGiven Point is: ",Question_Point)
    print("Closest Centroid is: ",Centroids[i])

def NormalizeData(df:pd.DataFrame):
    return (df-df.min())/(df.max()-df.min())