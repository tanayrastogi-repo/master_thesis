# -*- coding: utf-8 -*-

# Python Libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

# Thesis Utility functions
from Thesis_funcs import Utils
u = Utils()


    
#%%# ------------------- Exploratory Data Analysis ---------------------------- #
# Reading the dataset
path = os.getcwd() + '\\Data'
u.read_labels(path, verbose = 1)     


#%%# ------------------- Training Data ---------------------------- #
print("\n ------------------ Reading Training Data -------------------")
# Using Session 1 of the ACS-F2 dataset for training, session 2 will be used for testing
session = 1

# All the instances of the appliances that we want to use to creating training dataset
instance_range = [0,1,2,3]
#instance_range = [0,1,2,3]
# These are the appliances we want to read for creating appliance models
app_to_read = [0,1,5,3,9]

# Fetching Training Data
Traning_Data  = u.read_data(app_to_read, session, instance_range, filter_data=True, linePlot=False, scatterPlot=False)
unq_train = list(Traning_Data.keys())
unq_train.sort()

#%%# ------------------- Testing Data ---------------------------- #
print("\n ------------------ Reading Test Data -------------------")
# Reading data
session = 1

# All the instances of the appliances that we want to use to creating test dataset
instance_range = [5]
# These are the appliances we want to test
app_to_read = [0,1,5,3,9]

# Fetching Test Data
Test_Data = u.read_data(app_to_read, session, instance_range, filter_data=False, linePlot=False, scatterPlot=False)
unq_test = list(Test_Data.keys())
unq_test.sort()

# Create timeseries aggregated data for testing
Agg = u.create_agg_data(unq_test, Test_Data, lineplot = False, scatterplot = False)


#%%# ------------------- Gaussian Modelling ---------------------------- #
print("\n ------------------ Modelling Data -------------------")

# Features to use during clustering of models
feature_to_use = ['power', 'reacPower']

# Running a Model Estimation algorithm for GMM to find optimal clusters
for applicance in unq_train:
    # Running Gaussian Mixutre Model for each appliance
    opt_cluster = u.gaussian_model_estimation(Traning_Data[applicance],feature_to_use, applicance, plot=False)
    print("Number of cluster for {} : \t {}".format(applicance, opt_cluster))

# Mean, Variance and Weights for all the clusters using GMM
# Because of the issue with the Model Estimation on the given dataset, we manually input the clusters here.
Models = {}
opt_cluster_manual = {"Coffee,":2 ,"Computer,":1, "Fridge,":1, "Kettle,":2, "Microwave,": 3}
for applicance in unq_train:
    GMM, Mean, Vars, Wght = u.gaussian_clustering(Traning_Data[applicance],feature_to_use, applicance, opt_cluster_manual[applicance], verbose = False, plot=False)
    Models[applicance] ={'Mean': Mean, 'Vars': Vars , 'Wght': Wght, 'Label': applicance}


#%%# ------------------- Model Aggregation ---------------------------- #
print("\n ------------------ Aggregating Models -------------------")
# Defines how many clusters will be added together to create super cluster
aggregation_level = 3  
plot = False
Merged_Model = u.merge_clusters(Models, aggregation_level, plot=plot)
if(plot):
    for app in Agg['Label'].cat.categories:
        plt.scatter(Agg['power'][Agg['Label'] == app], Agg['reacPower'][Agg['Label'] == app], s=10, c='b', label=app)

#%%# ------------------- Testing Models ---------------------------- #
print('-------------------Model Test---------------------------')
# Ground Trueth
Y_true = u.getGNDtruth(Agg)
# Predictions
Y_pred = u.prediction(Agg, feature_to_use, Merged_Model)

#%%# ------------------- Evaluation ---------------------------- #
print('------------------- Accuracy Metrics ---------------------------')
classes = pd.Series(list(Y_true.cat.categories))
classes = classes.append(pd.Series(list(Y_pred.cat.categories)))
classes.sort_values(inplace=True)
classes.drop_duplicates(inplace=True)
classes = classes.reset_index(drop=True)

# Confusion matrix
cm = u.calculate_confusion_matrix(Y_true, Y_pred, labels=classes, plot=False)

# F-score 
u.f_score(Y_true, Y_pred, unq_test)


#%%# ------------------- Visualization ---------------------------- #
fig, ax = plt.subplots()
ax.set(xlim=(-10, len(Agg)+10), ylim=(-100, max(Agg['power']) + 200))
# Font types for printing
font_true = {'family': 'serif',
             'color':  'green',
             'weight': 'normal',
             'size': 16,
             }
font_false = {'family': 'serif',
             'color':  'darkred',
             'weight': 'normal',
             'size': 16,
             }

# Line plot
ax.plot(Agg.index.values, Agg['power'], 'r')
ax.set_ylabel('Real Power [Watt]')
ax.set_xlabel('Time [sec]')

# Scatter point
point = ax.scatter(Agg.index.values[0], Agg['power'][0], s=50, c='b')

def animate(i):  
    if(Y_true[i] == Y_pred[i]):
        ax.set_title('True: {} | Pred: {}'.format(Y_true[i], Y_pred[i]), fontdict = font_true)
    else:
        ax.set_title('True: {} | Pred: {}'.format(Y_true[i], Y_pred[i]), fontdict = font_false)
    point.set_offsets((Agg.index.values[i], Agg['power'][i]))
    
    return ax,
    
anmi = FuncAnimation(fig, animate, interval=70, frames=len(Agg.index.values)-1, blit = True)

anmi.save("Plot.mp4")

plt.show()
    