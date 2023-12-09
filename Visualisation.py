import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random 

def data_cluster_visualization(x,y,Point_Clusters,pandas_data:pd.DataFrame):


    for Cluster in Point_Clusters.values():
        arr = pd.DataFrame(Cluster)
        x_column = arr[:][x]
        y_column = arr[:][y]

        plt.scatter(x_column,y_column)      

    plt.xlabel(pandas_data.columns[x])
    plt.ylabel(pandas_data.columns[y])
    plt.show()
  
    
    return