from K_means import *
from Visualisation import *
from pandas import read_csv
import customtkinter as ctk

data = read_csv("midtermProject-part2-data.csv")
data = NormalizeData(data)

column_labels = data.columns.tolist()

x_column = 0
y_column = 1

def XChanged(val):
    global x_column
    x_column = column_labels.index(val)

def YChanged(val):
    global y_column
    y_column = column_labels.index(val)

def Calculate_and_Visualize():
    
    # run algorithm
    kval = K_value_Entry_text.get()
    epochval=Epoch_Entry_text.get()
    Centroids,Point_Clusters,Point_indexes = CalculateKmeans(data,k=kval,epoch=epochval)


    # write into text file
    datatowrite = ""
    for index,cluster in enumerate(Point_Clusters.values()):
        datatowrite += f"\nCluster {index+1}\n----------\n"
        for _,index_on_data in enumerate(Point_indexes[index]):
            datatowrite += f"\tRecord {index_on_data}\n"

    datatowrite += "\n\n"
    for index,cluster in enumerate(Point_Clusters.values()):
        datatowrite+= f"Cluster {index+1}: {len(cluster)} records\n"

    datatowrite += "\n\n"
    datatowrite += f"WCSS: {CalculateWCSS(Point_Clusters,Centroids)}\n"  # wcss smaller better
    datatowrite += f"BCSS: {CalculateBCSS(Point_Clusters,Centroids)}\n"  # bcss higher better
    datatowrite += f"Dunn Index: {CalculateDunnIndex(Point_Clusters,Centroids)}\n"   # dunn index higher better

    f = open("result.txt","w")
    f.writelines(datatowrite)
    f.close()

    # plot
    data_cluster_visualization(x_column,y_column,Point_Clusters,data)







#           GUI

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.geometry("500x500")
root.title("K Mean Algorithm")

K_value_Entry_text = ctk.IntVar(value=3)
Epoch_Entry_text = ctk.IntVar(value=1000)

frame = ctk.CTkFrame(master=root)
frame.pack(pady = 20,padx=60,fill="both",expand=True)

#               Parameter Entry
Kentry_label = ctk.CTkLabel(master=frame,text="K value")
Kentry_label.pack(pady=0,padx=10)
Kentry = ctk.CTkEntry(master= frame,placeholder_text="default 3",textvariable=K_value_Entry_text)
Kentry.pack(pady=0,padx=10)

Eentry_label = ctk.CTkLabel(master=frame,text="Epoch count")
Eentry_label.pack(pady=12,padx=10)
Eentry = ctk.CTkEntry(master= frame,textvariable=Epoch_Entry_text)
Eentry.pack(pady=0,padx=10)

#               Options
optionx_label = ctk.CTkLabel(master=frame,text="X value")
optionx_label.pack(pady=12,padx=10)
optionx = ctk.CTkOptionMenu(master = frame,values=column_labels,command=XChanged)
optionx.pack(pady=0,padx=10)
optionx.set(column_labels[0])

optiony_label = ctk.CTkLabel(master=frame,text="Y value")
optiony_label.pack(pady=12,padx=10)
optiony = ctk.CTkOptionMenu(master = frame,values=column_labels,command=YChanged)
optiony.pack(pady=0,padx=10)
optiony.set(column_labels[1])


visualize_Button = ctk.CTkButton(master=frame,text="Calculate and Visualize",command=Calculate_and_Visualize)
visualize_Button.pack(pady=30,padx=10)



root.mainloop()


