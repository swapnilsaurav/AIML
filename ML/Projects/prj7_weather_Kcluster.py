#Project 7 : Minute Weather Data Clustering using K-Means Clustering
#We will learn how to perform k-means lustering using scikit-learn in Python.

#Importing the Necessary Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
#Creating a Pandas DataFrame from a CSV file
data = pd.read_csv('data/weather.zip',compression = 'zip')

print("Data Shape: ",data.shape)
print("Sample Data: \n",data.head())
#########

###Data Sampling
#Lots of rows, so let us sample down by taking every 10th row.
sampled_df = data[(data['rowID'] % 10) == 0]
print("Sample Data Shape: ",sampled_df.shape)

#Statistics
print("1. ",sampled_df.describe().transpose())
print("2. ",sampled_df[sampled_df['rain_accumulation'] == 0].shape)
print("3. ",sampled_df[sampled_df['rain_duration'] == 0].shape)

#Drop all the Rows with Empty rain_duration and rain_accumulation
del sampled_df['rain_accumulation']
del sampled_df['rain_duration']
rows_before = sampled_df.shape[0]
sampled_df = sampled_df.dropna()
rows_after = sampled_df.shape[0]

#How many rows did we drop ?
print("How many rows did we drop ? ",rows_before - rows_after)
print("Columns in Sample Data: ",sampled_df.columns)

#Select Features of Interest for Clustering
features = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction',
        'max_wind_speed','relative_humidity']
select_df = sampled_df[features]
select_df.columns
select_df

#Scale the Features using StandardScaler
X = StandardScaler().fit_transform(select_df)
X

#Use k-Means Clustering
kmeans = KMeans(n_clusters=12)
model = kmeans.fit(X)
print("model\n", model)

#What are the centers of 12 clusters we formed ?
centers = model.cluster_centers_
print("centers: ", centers)
###Plots
#Let us first create some utility functions which will help us in plotting graphs:
# Function that creates a DataFrame with a column for Cluster Number

def pd_centers(featuresUsed, centers):
    colNames = list(featuresUsed)
    colNames.append('prediction')

    # Zip with a column called 'prediction' (index)
    Z = [np.append(A, index) for index, A in enumerate(centers)]

    # Convert to pandas data frame for plotting
    P = pd.DataFrame(Z, columns=colNames)
    P['prediction'] = P['prediction'].astype(int)
    return P

# Function that creates Parallel Plots

def parallel_plot(data):
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
    plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
    parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
    plt.show()

P = pd_centers(features, centers)
print("pd_centers: ",P)


#########
###Dry Days
parallel_plot(P[P['relative_humidity'] < -0.5])

###Warm Days
parallel_plot(P[P['air_temp'] > 0.5])

###Cool Days
parallel_plot(P[(P['relative_humidity'] > 0.5) & (P['air_temp'] < 0.5)])