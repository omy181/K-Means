import pandas as pd
import numpy as np
import math as math

def FindDistanceBetweenPoints(point1,point2):
    varsum = 0
    for c_var,p_var in zip(point1,point2):
        varsum += pow(c_var-p_var,2)
    distance = math.sqrt(varsum)
    return distance

def FindClosestCentroid(centroids,point):
    closest_distance = 0
    closest_centroid_index = 0

    for i,centroid in enumerate(centroids):
        #calculate distances
        distance = FindDistanceBetweenPoints(centroid,point)

        if(closest_distance == 0 or distance < closest_distance):
            closest_distance = distance
            closest_centroid_index = i

    return closest_centroid_index

def AssigntoNearestCentroid(centroids,points):
    Point_Clusters = {}
    Point_Indexes = {}
    for i in range(len(centroids)): 
        Point_Clusters[i] = []
        Point_Indexes[i] = []

    for i,point in enumerate(points):
        Closest_Centroid_index = FindClosestCentroid(centroids,point)
        Point_Clusters[Closest_Centroid_index].append(point)
        Point_Indexes[Closest_Centroid_index].append(i)
    
    return Point_Clusters,Point_Indexes

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
        Point_Clusters,_ = AssigntoNearestCentroid(centroids,points)
        centroids = FindNewCentroids(Point_Clusters)

    Point_Clusters,Point_indexes = AssigntoNearestCentroid(centroids,points)
    return centroids,Point_Clusters,Point_indexes


def NormalizeData(df:pd.DataFrame):
    return (df-df.min())/(df.max()-df.min())

def CalculateWCSS(Point_Clusters,Centroids):
    wcss = 0
    for cluster,centroid in zip(Point_Clusters.values(),Centroids):
        for point in cluster:
            distance = FindDistanceBetweenPoints(point,centroid)
            wcss += pow(distance,2)
    return wcss

def CalculateBCSS(Point_Clusters,Centroids):
    bcss = 0
    MiddleCentroid = sum(Centroids)/len(Centroids)

    for centroid,cluster in zip(Centroids,Point_Clusters.values()):
        bcss += pow(FindDistanceBetweenPoints(centroid,MiddleCentroid),2) * len(cluster)

    return bcss

def CalculateDunnIndex(Point_Clusters,Centroids):
    Dunn = 0

    # calculate the smallest distance between centroids
    smallest_centroid_dis = 0
    for index,centroid in enumerate(Centroids):
        if index+1 == len(Centroids) : break

        for _,centroid2 in enumerate(Centroids,start=index+1):
            dis = FindDistanceBetweenPoints(centroid,centroid2)
            if smallest_centroid_dis == 0 or smallest_centroid_dis > dis:
                smallest_centroid_dis = dis

    # calculate the highest distance between any 2 points in any cluster
    highest_point_dis = 0
    for index,cluster in enumerate(Point_Clusters.values()):
        if index+1 == len(Point_Clusters.values()) : break        
        for point in cluster:
            for _,point2 in enumerate(cluster,start=index+1):
                dis = FindDistanceBetweenPoints(point,point2)
                if highest_point_dis < dis:
                    highest_point_dis = dis

    Dunn = smallest_centroid_dis/highest_point_dis

    return Dunn