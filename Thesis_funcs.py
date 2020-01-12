# -*- coding: utf-8 -*-

# Python packages
import os
import glob 
import random
import numpy as np
import itertools

# EDA Packages
import pandas as pd
import xml.etree.ElementTree as ET

# Plotting and Visualization Packages
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn')

# Machine Learning packages
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal 
from sklearn.metrics import confusion_matrix

# Just to supress warning during printing
import warnings
warnings.filterwarnings('ignore')

# The dictonary to create shorthand names for all the appliances in th dataset
short_names = {'Coffee machines': 'Coffee,',
               'Computers stations (with monitors)': 'Computer,',
               'Fans': 'Fan,',
               'Fridges and freezers' : 'Fridge,',
               'Hi-Fi systems (with CD players)': 'HiFi,',
               'Kettles': 'Kettle,',
               'Lamps (compact fluorescent)': 'Lamp1,',
               'Lamps (incandescent)': 'Lamp2,',
               'Laptops (via chargers)': 'Laptop,',
               'Microwave ovens': 'Microwave,',
               'Mobile phones (via chargers)': 'Mobile,',
               'Monitors' :'Monitor,',
               'Printers' : 'Printer,',
               'Shavers (via chargers)': 'Shaver,',
               'Televisions (LCD or LED)': 'Tele,'}




class Utils:
    def __init__(self, ):
        self.path = None
        self.appliance_list = dict()
        self.unq_appliance = pd.DataFrame()
        pass
    
    def read_labels(self, path, verbose=0):
        """
        The function reads the appliance data in two differnt session of ACS-F2 dataset.
        Input:
            path: The folder path of the data
        Return:
            Appliance: A dictonary with all the appliance data for both sessions
        """
        # Dataset Path
        self.path = path
        
        # Find all the appliances name in the dataset
        appliances_category = os.listdir(self.path)
        appliances_category.sort()
        
        # For each appliance, find the all instances of appliance
        for app in appliances_category:
            app_path = path + '/'+ app
            # Create a list of all the appliance by sessions
            Session = {}
            # Running for the session 1 and 2 in the range function
            for i in range(1,3):
                App = []
                os.chdir(app_path)
                for file in glob.glob('*a{}.xml'.format(i)):
                    App.append(file)
                App.sort()
                Session[i] = App
        
            # Update the Dict for all the appliances and instances with sessions
            self.appliance_list[app] = Session
        
        # Create a dataframe for all the unique appliances in the dataset            
        self.unq_appliance = pd.DataFrame(list(self.appliance_list.keys()), columns=['Appliances'])
        self.unq_appliance.sort_values(by=['Appliances'], inplace=True)
        self.unq_appliance.reset_index(drop=True, inplace=True)
            
        # Print Summary
        if(verbose > 0):
            print(" ---------- Appliances in the Dataset ------------ ")
            print(self.unq_appliance)
        if(verbose > 1):
            print(" ---------- Number of Appliances in the Dataset ------------ ")
            df = pd.DataFrame(index = self.unq_appliance['Appliances'].to_list())
            for key in df.index:
                df.at[key, "Sess: 1 "] = len(self.appliance_list[key][1])
                df.at[key, "Sess: 2 "] = len(self.appliance_list[key][2])
            print(df) 
            print('\n')
    


    def read_appliance_specific_data(self, appliance, session, instance_range, filter_data = False, plot = False):
        """
        Function to read appliance instance for specific appliance
        Input:
            appliance:      Specific appliances to read
            session:        Session of data
            instance_range: Instances of appliances in the dataset
            filter_data:    Filter the median filter on the data 
            linePlot:       Line plot for all the appliances instance data
        Return:
            data:           A dictonary of instance data for the selected appliances for particular session
        """
        
        # Empty variable for data
        data = pd.DataFrame()
        
        # Features to drop
        drop_features = ['rmsVolt','time','freq']               
        
        # Plotting Setup
        if(plot):
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            fig.suptitle("{}".format(appliance), fontsize=16)
            
        if(filter_data):
            if(plot):
                fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True)
                fig2.suptitle("Data without GND",fontsize=16)
            
        # Append all the data for all the instances
        for instance in instance_range:
            
            # File path to be read
            path_instance = self.path + '/' + appliance + '/' + self.appliance_list[appliance][session][instance]
            
            # Check if the path exits
            if(os.path.exists(path_instance)):                
                print('Reading Instance {} for {}'.format(instance, appliance))
                
                # Data at the instance
                instance_data = pd.DataFrame()
                
                # Read the xml file
                tree = ET.parse(path_instance)
                root = tree.getroot()
                           
                # Extract Data for the appliance
                signalCurve = root[5]
                for signalPoint in signalCurve:
                    point_data = pd.DataFrame.from_dict(signalPoint.attrib, orient='index').transpose()
                    instance_data = instance_data.append(point_data)
                
                # Reset the index
                instance_data.reset_index(drop=True, inplace=True)
                # Convert the data type for the coloumns
                instance_data = instance_data.convert_objects(convert_dates=True, convert_numeric=True)
                # Drop the features that are not required
                instance_data = instance_data.drop(drop_features, axis = 1)
                
                # Plotting Line plot for appliance values
                if(plot):
                    # Plotting
                    ax[0].plot(instance_data.index.values, instance_data['power'], label = 'App{}'.format(instance))
                    ax[0].set_ylabel('P [W]')
                    
                    ax[1].plot(instance_data.index.values, instance_data['reacPower'], label = 'App{}'.format(instance))
                    ax[1].set_ylabel('Q [Var]')
                                                   
                    ax[0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.0), shadow=True)
                
                # Median Filter to the data to remove GND values            
                if(filter_data):                   
                    threshold = instance_data['power'].mean()
                    if(appliance == 'Microwave ovens'):
                        threshold = instance_data['power'][instance_data ['power'] < instance_data['power'].mean()].mean()
                    
                    instance_data = instance_data[instance_data['power'] > threshold]
                    # Median Filter
                    instance_data = instance_data.rolling(window=3).median()
                    instance_data = instance_data.dropna(axis = 0, how = 'any')
    
                    
                data = data.append(instance_data)
                data.reset_index(drop=True, inplace=True)
                
                # Introduce labels to the data
                data['Label'] = pd.Series([short_names[appliance]]*data.shape[0])
                data['Label'] = data['Label'].astype('str')
           
            # If not raise error
            else:
                raise ValueError('The path does not exit')
        
        
        return data






    def read_data(self, app_to_read, session, instance_range, filter_data = False, linePlot=False, scatterPlot = False):
        """
        The function reads the data from the dataset for all the selected appliances for a particular session
        Input:
            app_to_read:    List of appliances to read
            session:        Session of data
            instance_range: Instances of appliances in the dataset
            filter_data:    Filter the median filter on the data 
            linePlot:       Line plot for all the appliances instance data
            scatterPlot:    Scatter plot for all the appliance data
        Return:
            Data:           A dictonary of instance data for all the selected appliances for particular session
        """
        Data = {}
        
        # Print Summry
        print(' ---- Reading data with filter {}\n'.format(filter_data))
        
        for uniqueappliance in self.unq_appliance.loc[app_to_read, 'Appliances']:
            temp = self.read_appliance_specific_data(uniqueappliance, session, instance_range, filter_data, linePlot)
            Data[short_names[uniqueappliance]] = temp
        
        if(scatterPlot):              
            keys = list(Data.keys())
            keys.sort()            
            fig, ax = plt.subplots()
#            ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
            
            for app in keys:
                ax.scatter(Data[app]['power'], Data[app]['reacPower'], label = app, s = 10)
            ax.legend()
            ax.set_title('Scatter plot')
            ax.set_xlim([-100, 2500])
            ax.set_ylim([-100,800])
            ax.set_xlabel('Real Power [W]')
            ax.set_ylabel('Reactive Power [Var]')

        # Print Summary
        print("\n ---------- Data Summary ------------ ")
        print("Appliances in the Data: ", list(Data.keys()))
        print("Instances for each appliances: #{}: {}".format(len(instance_range), list(instance_range)))
        print("Length of Dataset for appliance")
        for key, val in Data.items():
            print("{} : {}".format(key, len(val)))
            
        return Data





    def single_data(self, unq, Data):
        """
        Randomly selet one appliance to add to aggregated data
        """
        rand = random.choice(unq)
        print(rand)
        df = Data[rand]
        return df
    
    def double_data(self, unq, Data):
        """
        Randomly selet two appliances to add to aggregated data
        """
        rand = random.sample(unq,2)
        print(rand)
        small = min(len(Data[rand[1]]), len(Data[rand[0]]))   
        df = Data[rand[0]][:small].add(Data[rand[1]][:small], axis='index', fill_value=True)
        return df
        
    def triple_data(self, unq, Data):
        """
        Randomly selet three appliances to add to aggregated data
        """
        rand = random.sample(unq,3)
        print(rand)
        small = min(len(Data[rand[2]]), len(Data[rand[1]]), len(Data[rand[0]]))
        temp = Data[rand[0]][:small].add(Data[rand[1]][:small], axis='index', fill_value=True)
        df =  Data[rand[2]][:small].add(temp, axis='index', fill_value=True)
        return df
    
    def GND(self, length):
        """
        Function to add GND signal to add to aggregated datas
        """
        GND = pd.DataFrame(np.zeros((length,4)), columns=['phAngle','rmsCur','power', 'reacPower'])
        GND['Label'] = pd.Series(['GND,']*GND.shape[0])
        GND['Label'] = GND['Label'].astype('str')
        return GND
    
    
    def phase_angle_calculate(self, df):
        df['phaseTest'] = np.sqrt(np.square(df['power']) + np.square(df['reacPower']))
        mask = df['phaseTest'] != 0
        df1 = np.degrees(np.arccos(df['power'][mask]/df['phaseTest'][mask]))
        df.loc[df1.index, 'phaseTest'] = df1
        df['phAngle'] = df['phaseTest']
        df.drop(['phaseTest'], inplace = True, axis = 1)
        return df
    
    
    def create_agg_data(self, unq, Data, lineplot = False, scatterplot = False):
        """
        Function to create aggregated time series for testing. 
        Inputs:
            unq:         Unique appliances in the test data
            Data:        Test Dataset
            lineplot:    Line plot for test dataset
            scatterplot: Scatterplot for test dataset
        Return:
            agg:    Dataframe with aggregated time series data
        """
        
        # Initial variable for the aggregated data
        agg = pd.DataFrame(np.zeros((1,4)), columns=['phAngle','rmsCur', 'power', 'reacPower'])
        agg['Label'] = pd.Series(['GND,']*agg.shape[0])
        agg['Label'] = agg['Label'].astype('str')
        
        print('\nThe aggregated time series contains...')
        
        # Single Data
        agg = pd.concat([agg, self.single_data(unq, Data)])
        # Add GND
        agg = pd.concat([agg, self.GND(50)])
        # Double Data    
        agg = pd.concat([agg, self.double_data(unq, Data)])
        # Add GND
        agg = pd.concat([agg, self.GND(50)])
        # Triple Data
        agg = pd.concat([agg, self.triple_data(unq, Data)])
        # Single Data
        agg = pd.concat([agg, self.single_data(unq, Data)])
        # Add GND
        agg = pd.concat([agg, self.GND(50)])
        # Double Data    
        agg = pd.concat([agg, self.double_data(unq, Data)])
        # Add GND
        agg = pd.concat([agg, self.GND(50)])
        # Triple Data
        agg = pd.concat([agg, self.triple_data(unq, Data)])
        # Add GND
        agg = pd.concat([agg, self.GND(50)])
        
        # Reset Index
        agg.reset_index(drop=True, inplace=True)    
        agg['Label'] = agg['Label'].astype('category')
        
        # Get the phase angle right
        agg = self.phase_angle_calculate(agg)
        
        # Ploting the aggregated data
        if(lineplot):
            _, line_ax = plt.subplots(2,1, sharex=True)        
            line_ax[0].plot(agg.index.values, agg['power'], 'r')
            line_ax[1].plot(agg.index.values, agg['reacPower'], 'b')
            
            line_ax[0].set_ylabel('Real Power [Watt]')
            line_ax[0].legend()
            
            line_ax[1].set_xlabel('Time [sec]')
            line_ax[1].set_ylabel('Reactive Power [Var]')
            line_ax[1].legend()
            
            line_ax[0].set_title('Plot for aggregated signal')
        
           
        if(scatterplot):
            _,  scatter_ax = plt.subplots()
            scatter_ax.scatter(agg['power'], agg['reacPower'], s=10)
            scatter_ax.set_xlabel('Real Power [Watt]')
            scatter_ax.set_ylabel('Reactive Power [Var]')
            scatter_ax.set_title('Plot for Aggregated Signal')
               
        return agg

    
    
    
    
    
    def gaussian_model_estimation(self, data, feature_to_use, name, plot=False):
        """
        Function to estimate the number of clusters to use for the GMM modelling
        Inputs:
            data:               Dataframe for modelling
            feature_to_use:     Features in the dataframe to use
            name:               Name of appliance (for plotting)
            plot:               Line plot for BIC values
        Return:
            opt_cluster:        Number of optimal clusters for the dataset
        """
        # Extracting data from teh Dataframe 
        data = data.as_matrix(columns=feature_to_use)
        
        if(plot):
            fig = plt.figure()
            ax = fig.add_subplot(111)    
            
        n_estimator = np.arange(1, 10)
        clfs = [GaussianMixture(n_components = n, covariance_type='full', init_params='kmeans', random_state = 5).fit(data) for n in n_estimator]
        bics = pd.DataFrame([clf.bic(data) for clf in clfs])
        
        # Finding the optimum clusters
        bics_shift = bics.shift(1) - bics
        opt_cluster = int(bics_shift.idxmax()) + 1
        
        if(plot):  
            ax.plot(n_estimator, bics)
            ax.set_title('GMM Model Estimation for {}'.format(name))
            ax.legend()
            ax.set_xlabel('Number of Clusters')
        
        return opt_cluster






    def draw_ellipse(self, position, covariance, ellipse_level = 3, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        
        # Draw x at means
        ax.scatter(position[0], position[1], s = 60, marker = '+', c='r', zorder = 3)
            
        # Draw the Ellipse
        for nsig in range(1, ellipse_level):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, lw = 0.5, ec = 'k', fc='none', ls = 'dashed'))
    
    def plot_gmm(self, gmm, data, label=True, ax=None):  
        """
        Function to plot scatter plot for the GMM models with an ellipse to show means and variance
        Inputs:
            gmm:                Appliance Model
            X:                  Dataframe for modelling
        """         
        ax = ax or plt.gca()        
        labels = gmm.fit(data).predict(data)
        if label:
            ax.scatter(data[:, 0], data[:, 1], c=labels, s=5, zorder=2, cmap='rainbow')
        else:
            ax.scatter(data[:, 0], data[:, 1], s=10, zorder=2)
        w_factor = 0.6 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            self.draw_ellipse(pos, covar, ax = ax, alpha=w * w_factor)
        
        
    def gaussian_clustering(self, data, feature_to_use, name, components, verbose = True, plot=False):
        """
        Function to model the given data with features using Gaussian Mixture Model (GMM)
        Inputs:
            data:               Dataframe for modelling
            feature_to_use:     Features in the dataframe to use
            name:               Name of appliance (for plotting)
            components:         Number of clusters to use for modelling
        Return:
            gmm:                Appliance Model
            gmm.means_:         Model Means
            gmm.covariance_:    Model variance
            gmm.weights_:       Model weights
        """
        # Extracting data from teh Dataframe 
        data = data.as_matrix(columns=feature_to_use)
        # Gaussing Mixture Model
        gmm = GaussianMixture(n_components = components, covariance_type='full', init_params='kmeans', random_state = 5).fit(data)
        
        if(verbose):
            # Print Summry of the GMM
            print('Gaussian Cluster for {}'.format(name))
            print('Mean: ',gmm.means_)
            print('Variance: ',gmm.covariances_)
            print('Weights: ', gmm.weights_)
            print('\n')
        
        # Plotting
        if (plot):
            fig = plt.figure()
            cluster_ax = fig.add_subplot(111)
            self.plot_gmm(gmm, data, label=True, ax=cluster_ax)  
            cluster_ax.set_ylim(-200, 800)
            cluster_ax.set_xlim(-200, 2500)
            cluster_ax.set_title('Gaussian Clusters for {}'.format(name))
            cluster_ax.set_xlabel('Real Power [Watt]')
            cluster_ax.set_ylabel('Reactive Power [Var]')
        
        return gmm, gmm.means_, gmm.covariances_, gmm.weights_




    def merge_clusters(self, Models, merge_level,  plot = False):
        """
        Function to merge the gaussian cluster to create super clusters. The merging happens based on the merge_level
        Inputs:
            Models:             Models for merging
            merge_level:        Model merge level. Determines how many models to merge
        Return:
            merged_Model:       Dictonary with all the merged models
        """
        
        # Appliances in the model
        unq_appliance = list(Models.keys())
        
        ## Merging Clusters to make Super clusters
        merge_mean = []
        merge_var = []
        merge_wght = []
        merge_name = []
        
        
        # Merge Level 2
        if(merge_level > 1):
            for sng1 in range(0, len(unq_appliance)):
                for sng2 in range(sng1+1, len(unq_appliance)):                                        
                    for i in range(0, Models[unq_appliance[sng1]]['Mean'].shape[0]):                                    
                        for j in range(0, Models[unq_appliance[sng2]]['Mean'].shape[0]):
                            merge_mean.append(Models[unq_appliance[sng1]]['Mean'][i] + Models[unq_appliance[sng2]]['Mean'][j])
                            merge_var.append(Models[unq_appliance[sng1]]['Vars'][i] + Models[unq_appliance[sng2]]['Vars'][j])
                            merge_wght.append(Models[unq_appliance[sng1]]['Wght'][i] * Models[unq_appliance[sng2]]['Wght'][j])
    #                        merge_name.append(Models[unq_appliance[sng1]]['Label'] + 'Clt{}'.format(i) + ' + ' + Models[unq_appliance[sng2]]['Label'] + 'Clt{}'.format(j) )     
                            merge_name.append(Models[unq_appliance[sng1]]['Label'] + Models[unq_appliance[sng2]]['Label'])     
                            
        # Merge Level 3
        if(merge_level > 2):
            for sng1 in range(0, len(unq_appliance)):
                for sng2 in range(sng1+1, len(unq_appliance)):
                    for sng3 in range(sng2+1, len(unq_appliance)):                    
                        for i in range(0, Models[unq_appliance[sng1]]['Mean'].shape[0]):
                            for j in range(0, Models[unq_appliance[sng2]]['Mean'].shape[0]):
                                for k in range(0, Models[unq_appliance[sng3]]['Mean'].shape[0]):
                                    merge_mean.append(Models[unq_appliance[sng1]]['Mean'][i] + Models[unq_appliance[sng2]]['Mean'][j] + Models[unq_appliance[sng3]]['Mean'][k])
                                    merge_var.append(Models[unq_appliance[sng1]]['Vars'][i] + Models[unq_appliance[sng2]]['Vars'][j] + Models[unq_appliance[sng3]]['Vars'][k])
                                    merge_wght.append(Models[unq_appliance[sng1]]['Wght'][i] * Models[unq_appliance[sng2]]['Wght'][j] * Models[unq_appliance[sng3]]['Wght'][k])
    #                                merge_name.append(Models[unq_appliance[sng1]]['Label'] + 'Clt{}'.format(i) +' + '+ Models[unq_appliance[sng2]]['Label'] + 'Clt{}'.format(j) + ' + ' + Models[unq_appliance[sng3]]['Label'] + 'Clt{}'.format(k))
                                    merge_name.append(Models[unq_appliance[sng1]]['Label'] + Models[unq_appliance[sng2]]['Label'] + Models[unq_appliance[sng3]]['Label'])
        
        
        # Merge Level 4
        if(merge_level > 3):
            for sng1 in range(0, len(unq_appliance)):
                for sng2 in range(sng1+1, len(unq_appliance)):
                    for sng3 in range(sng2+1, len(unq_appliance)):
                        for sng4 in range(sng3+1, len(unq_appliance)):
                            for i in range(0, Models[unq_appliance[sng1]]['Mean'].shape[0]):
                                for j in range(0, Models[unq_appliance[sng2]]['Mean'].shape[0]):
                                    for k in range(0, Models[unq_appliance[sng3]]['Mean'].shape[0]):
                                        for l in range(0, Models[unq_appliance[sng4]]['Mean'].shape[0]):
                                            merge_mean.append(Models[unq_appliance[sng1]]['Mean'][i] + Models[unq_appliance[sng2]]['Mean'][j] + Models[unq_appliance[sng3]]['Mean'][k] + Models[unq_appliance[sng4]]['Mean'][l])
                                            merge_var.append(Models[unq_appliance[sng1]]['Vars'][i] + Models[unq_appliance[sng2]]['Vars'][j] + Models[unq_appliance[sng3]]['Vars'][k] + Models[unq_appliance[sng4]]['Vars'][l])
                                            merge_wght.append(Models[unq_appliance[sng1]]['Wght'][i] * Models[unq_appliance[sng2]]['Wght'][j] * Models[unq_appliance[sng3]]['Wght'][k] * Models[unq_appliance[sng4]]['Wght'][l])
    #                                        merge_name.append(Models[unq_appliance[sng1]]['Label'] + 'Clt{}'.format(i) + ' + ' + Models[unq_appliance[sng2]]['Label'] + 'Clt{}'.format(j) +' + '+ Models[unq_appliance[sng3]]['Label'] + 'Clt{}'.format(k) +' + '+ Models[unq_appliance[sng4]]['Label'] + 'Clt{}'.format(l))
                                            merge_name.append(Models[unq_appliance[sng1]]['Label'] + Models[unq_appliance[sng2]]['Label'] + Models[unq_appliance[sng3]]['Label'] + Models[unq_appliance[sng4]]['Label'])
        
        
        
        
        # Add original cluster to the lists
        for i in unq_appliance:
            for j in range(0, Models[i]['Mean'].shape[0]):
                merge_mean.append(Models[i]['Mean'][j])
                merge_var.append(Models[i]['Vars'][j])
                merge_wght.append(Models[i]['Wght'][j])
                merge_name.append(i)
        
        
        # Convert to Numpy array just for plotting
        merge_mean = np.array(merge_mean)
        merge_var = np.array(merge_var)
        merge_wght = np.array(merge_wght) 
        merge_name = np.array(merge_name)
        
        number_of_cluster = len(merge_mean)
        
        
        if(plot):        
            ## Plotting the super clusters
            fig, ax = plt.subplots() 
            w_factor = 0.6 / merge_wght.max()
            for pos, covar, w, name in zip(merge_mean, merge_var, merge_wght, merge_name):
                self.draw_ellipse(pos, covar, ellipse_level = 2, ax = ax, alpha=w * w_factor)
    #            ax.text(pos[0], pos[1], name, fontsize=6, color='blue')
            ax.set_title('Number of Super Clusters: {}'.format(number_of_cluster)) 
            ax.set_xlabel('Real Power [Watt]')
            ax.set_ylabel('Reactive Power [Var]')
            ax.set_ylim(-100, 1000)
            ax.set_xlim(-100, 4000)
        
        
        # Pass the merged clusters as a dictonary
        merged_Model = {}
        for i in range(0, len(merge_mean)):
            merged_Model[i] = {'Mean': merge_mean[i], 'Vars': merge_var[i], 'Wght' : merge_wght[i] , 'Label' : merge_name[i]}
        
        
           
        # Add GND as a model in the clusters
        num_model = len(merged_Model[0]['Mean'])
        merged_Model[len(merged_Model)] = {'Mean': np.ones((1,num_model)).flatten(), 'Vars': np.eye(num_model), 'Wght' : 1.0 , 'Label' : 'GND,'}
        
        
        print('\n Number of merged clusters: {}'.format(len(merged_Model)))
        return merged_Model




    def split(self, x):
        x = x.split(',')
        x.sort()
        return ''.join(x)
    
    def getGNDtruth(self, Agg):
        """
        Funciton to get true values from the aggregated data at each time step
        Inputs:
            Agg:    Aggregated Data
        Return 
            Dataframe for the true values
        """
        x = [self.split(pd.Series(Agg['Label'])[i]) for i in pd.Series(Agg['Label']).index]
        return pd.Series(x, dtype = 'category')
    
    
    def prediction(self, Agg, feature_to_use, Merged_Model):
        """
        Function to predict the label for each datapoint of the aggregated data.
        We run a Maximum Likelihood Estimation (MLE) to predict the labels for each datapoint.
        Inputs:
            Agg:                Aggregated Data
            feature_to_use:     Features in the test data for predictions
            Merged_Model:       GMM Super cluster of the appliances.
        Return 
            Dataframe for the true values
        """
        # Extract only the power data fromm the Dataframe
        data = Agg.as_matrix(columns=feature_to_use)
        # List variable for the identified names from AGG     
        signal_identified = []
        
        # Resource matrix   
        r = np.zeros((len(data), len(Merged_Model)))
        
        print('Length of the Aggregated signal', len(data))
        
        for n in range(len(data)):
            for k in Merged_Model:
                 m = Merged_Model[k]['Mean']
                 v = Merged_Model[k]['Vars']
                 w = Merged_Model[k]['Wght']
                 
                 if (np.linalg.det(Merged_Model[k]['Vars']) < 1.0):
                     v += 0.001
                 
                 r[n][k] = w*multivariate_normal.pdf(data[n], mean=m, cov=v)
         
            # Normalize to get the probabilities   
            r[n] = np.round(r[n]/sum(r[n]), 2)
            
            # Identify the label for the datapoint
            prediction = self.split(Merged_Model[np.argmax(r[n])]['Label'])
            if n%100 == 0:
                print("Prediction for the datapoint {}: {}".format(n, prediction))
            signal_identified.append(prediction)
       
        Y_pred = pd.Series(signal_identified, dtype='category')
        return Y_pred



    def calculate_confusion_matrix(self, Y_true, Y_pred, labels=None, plot=False):
        """
        Function to calculate confusion matrix for the predictions and ground truth
        Inputs:
            Y_true:     True labels/ Ground truth
            Y_pred:     Predictions for each time step
            labels:     All possible labels for ground truth
            plot:       Plotting Confusion Map
        Return 
            Confusion matrix
        """
        # Calculate the Confusion matrix for the True, Prediction pair
        cm = confusion_matrix(Y_true, Y_pred, labels=labels)
        
        if(plot):
            gnd_truth = list(Y_true.cat.categories)
            color = ['green' if label in gnd_truth else 'red' for label in labels]
            
            title='Confusion matrix'
            cmap=plt.cm.Blues
            
            fig, ax = plt.subplots()
            ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.set_title(title)
            tick_marks = np.arange(len(labels))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(labels)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(labels)
            
            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                ax.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            
            for xtick, ytick, color in zip(ax.get_xticklabels(), ax.get_yticklabels(), color):
                xtick.set_color(color)
                ytick.set_color(color)
        
            fig.tight_layout()
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
            plt.xticks(rotation=90)
        
        return cm

    def f_score(self, Y_true, Y_pred, unq_test):
        """
        Function to calculate F_score for predictions and ground truth
        Inputs:
            Y_true:     True labels/ Ground truth
            Y_pred:     Predictions for each time step
            unq_test:   Labels from test data
        Return 
            F-score for all the appliances in the dataset
        """
        unq = [self.split(unq_test[i]) for i in range(0,len(unq_test))]
        for sng in unq:
            true = [1 if sng in i else 0 for i in Y_true]
            pred = [1 if sng in i else 0 for i in Y_pred]
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f_score = 2*precision*recall/(precision+recall)
            accuracy = (tn+tp)/(tn+tp+fn+fp)
            
    #        print('True positive for {} : {}'.format(sng, tp))
    #        print('True negative for {} : {}'.format(sng, tn))
    #        print('False positive for {} : {}'.format(sng, fp))
    #        print('False negative for {} : {}'.format(sng, fn))
            print(" \n------ Evaluation for {}".format(sng))
            print('Precision :{}'.format(precision))
            print('Recall    :{}'.format(recall))
            print('Accuracy  :{}'.format(accuracy))
            print('F_score   :{}'.format(f_score))

    def animated_plots(self, Agg, Y_true, Y_pred, ):
        
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
        FuncAnimation(fig, animate, interval=70, frames=len(Agg.index.values)-1, blit = True)
        
        plt.draw()
        