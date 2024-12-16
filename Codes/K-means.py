# -*- coding: utf-8 -*-
"""
Ana P. O. Costa et. al. "Manufacturing process encoding through natural language processing for prediction of material properties"
2023
"""

# Import necessary libraries
import csv
import numpy as np
import pandas as pd
import os
import itertools

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Set the device to use GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read steel_n10.csv file into a dataframe
dataset_df = pd.read_csv("steel_database.csv", sep=';', low_memory=False)

# Create a dataframe with features for training 
#dataframe 1 - Cluster by manufacturing processing
x = pd.DataFrame(dataset_df, columns=['Cr', 'Al', 'B', 'Co', 'Mo', 'Ni', 'Ti', 'Zr', 'C', 'Fe', 'Pd',
                                            'Mn', 'P', 'Si', 'N', 'Cu', 'Nb', 'Se', 'Ta', 'W', 'V', 'S', 
                                            'YS', 'UTS', 'Elongation', 'HB'])

#dataframe 2 - Cluster by mechanical properties
#x = pd.DataFrame(dataset_df, columns=['Cr', 'Al', 'B', 'Co', 'Mo', 'Ni', 'Ti', 'Zr', 'C', 'Fe', 'Pd',
#                                           'Mn', 'P', 'Si', 'N', 'Cu', 'Nb', 'Se', 'Ta', 'W', 'V', 'S',
#                                             'e1',	'e2',	'e3',	'e4','e5', 'e6', 'e7',	'e8',	'e9',	
#                                                'e10', 'e11', 'e12','e13','e14', 'e15',
#                                             'e16',	'e17',	'e18',	'e19', 'e20', 'e21','e22','e23',
#                                                   'e24',	'e25',	'e26',	'e27',	'e28', 'e29','e30','e31',
#                                                  'e32', 'e33',	'e34',	'e35',	'e36',	'e37', 'e38', 'e39',
#                                                   'e40', 'e41', 'e42',	'e43',	'e44',	'e45',	'e46', 'e47',
#                                                   'e48','e49','e50', 'e51',	'e52',	'e53',	'e54',	'e55', 'e56',
#                                                   'e57','e58','e59', 'e60',	'e61',	'e62',	'e63',	'e64', 'e65',
#                                                  'e66','e67','e68', 'e69',	'e70',	'e71',	'e72',	'e73', 'e74',
#                                                   'e75','e76','e77', 'e78',	'e79',	'e80',	'e81',	'e82', 'e83',
#                                                   'e84','e85','e86', 'e87',	'e88',	'e89',	'e90',	'e91', 'e92',
#                                                    'e93','e94','e95', 'e96',	'e97',	'e98',	'e99',	'e100', 'e101',
#                                                   'e102','e103','e10', 'e105', 'e106','e107','e108', 'e109', 'e110',
#                                                  'e111','e112','e113', 'e114','e115',	'e116',	'e117',	'e118',	'e119',
#                                                  'e120', 'e121','e122', 'e123', 'e124', 'e125', 'e126', 'e127',
#                                                 'e128', 'e129', 'e130', 'e131','e132', 'e133', 'e134', 'e135','e136', 'e136', 'e137'
#                                                 'e135',	'e136',	'e137',	'e138',	'e139',	'e140',	'e141',	'e142',	'e143',	'e144',
#                                                 'e145',	'e146',	'e147',	'e148',	'e149',	'e150',	'e151',	'e152',	'e153',	'e154',
#                                                 'e155',	'e156',	'e157',	'e158',	'e159',	'e160',	'e161',	'e162',	'e163',	'e164',
#                                                 'e165',	'e166',	'e167',	'e168',	'e169',	'e170',	'e171',	'e172',	'e173',	'e174',
#                                                 'e175',	'e176',	'e177',	'e178',	'e179',	'e180',	'e181',	'e182',	'e183',	'e184',
#                                                 'e185',	'e186',	'e187',	'e188',	'e189',	'e190',	'e191',	'e192',	'e193',	'e194',
#                                                 'e195',	'e196',	'e197',	'e198',	'e199',	'e200',	'e201',	'e202',	'e203',	'e204'])

#dataframe 3 - Cluster by Cluster by Chemical composition (C, Mn, Ni and Cr)
#x = pd.DataFrame(dataset_df, columns=['e1',	'e2',	'e3',	'e4','e5', 'e6', 'e7',	'e8',	'e9',	
#                                                'e10', 'e11', 'e12','e13','e14', 'e15',
#                                             'e16',	'e17',	'e18',	'e19', 'e20', 'e21','e22','e23',
#                                                   'e24',	'e25',	'e26',	'e27',	'e28', 'e29','e30','e31',
#                                                  'e32', 'e33',	'e34',	'e35',	'e36',	'e37', 'e38', 'e39',
#                                                   'e40', 'e41', 'e42',	'e43',	'e44',	'e45',	'e46', 'e47',
#                                                   'e48','e49','e50', 'e51',	'e52',	'e53',	'e54',	'e55', 'e56',
#                                                   'e57','e58','e59', 'e60',	'e61',	'e62',	'e63',	'e64', 'e65',
#                                                  'e66','e67','e68', 'e69',	'e70',	'e71',	'e72',	'e73', 'e74',
#                                                   'e75','e76','e77', 'e78',	'e79',	'e80',	'e81',	'e82', 'e83',
#                                                   'e84','e85','e86', 'e87',	'e88',	'e89',	'e90',	'e91', 'e92',
#                                                    'e93','e94','e95', 'e96',	'e97',	'e98',	'e99',	'e100', 'e101',
#                                                   'e102','e103','e10', 'e105', 'e106','e107','e108', 'e109', 'e110',
#                                                  'e111','e112','e113', 'e114','e115',	'e116',	'e117',	'e118',	'e119',
#                                                  'e120', 'e121','e122', 'e123', 'e124', 'e125', 'e126', 'e127',
#                                                 'e128', 'e129', 'e130', 'e131','e132', 'e133', 'e134', 'e135','e136', 'e136', 'e137'
#                                                 'e135',	'e136',	'e137',	'e138',	'e139',	'e140',	'e141',	'e142',	'e143',	'e144',
#                                                 'e145',	'e146',	'e147',	'e148',	'e149',	'e150',	'e151',	'e152',	'e153',	'e154',
#                                                 'e155',	'e156',	'e157',	'e158',	'e159',	'e160',	'e161',	'e162',	'e163',	'e164',
#                                                 'e165',	'e166',	'e167',	'e168',	'e169',	'e170',	'e171',	'e172',	'e173',	'e174',
#                                                 'e175',	'e176',	'e177',	'e178',	'e179',	'e180',	'e181',	'e182',	'e183',	'e184',
#                                                 'e185',	'e186',	'e187',	'e188',	'e189',	'e190',	'e191',	'e192',	'e193',	'e194',
#                                                 'e195',	'e196',	'e197',	'e198',	'e199',	'e200',	'e201',	'e202',	'e203',	'e204',
#                                                  'YS', 'UTS', 'Elongation', 'HB'])
# Convert to float32 type
x = x.astype(np.float32)

# Replace NaN values with zero
x = x.replace(np.nan, 0)

# Save the dataframe to a CSV file
x.to_csv('x.csv', sep=';', index=False)

# Convert dataframe to a numpy array
x = x.values


# Read manufacturing.csv file into a dataframe
#dataframe 1
dataset_df = pd.read_csv("manufacturing.csv", sep=';', low_memory=False)

# Read mechanical.csv file into a dataframe
#dataframe 2
#dataset_df = pd.read_csv("mechanical.csv", sep=';', low_memory=False)

# Read chemical.csv file into a dataframe
#dataframe 3
#dataset_df = pd.read_csv("chemical.csv", sep=';', low_memory=False)



# Create a dataframe with the features for training
y = pd.DataFrame(dataset_df, columns=['name'])

# Convert to string type
y = y.astype(str)

# Replace NaN values with zero
y = y.replace(np.nan, 0)

# Save the dataframe to a CSV file
y.to_csv('y.csv', sep=';', index=False)

# Convert dataframe to a numpy array
#y = y.values
 
# Use label encoding to convert string labels to numerical values
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

y[:5]
label_encoder.classes_

# Number of clusters is determined by the number of unique classes in the target variable
n_clusters = len(label_encoder.classes_)

# Define the preprocessing pipeline including MinMax scaling and PCA for dimensionality reduction
preprocessor = Pipeline([
    ("scaler", MinMaxScaler()),
    ("pca", PCA(n_components=2, random_state=42)),
])

# Define the clustering pipeline using KMeans
clusterer = Pipeline([
    ("kmeans", KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=50,
        max_iter=500,
        random_state=42,
    )),
])

# Create the main pipeline combining preprocessing and clustering
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("clusterer", clusterer)
])


# Fit the pipeline on the data
pipe.fit(x)
preprocessed_data = pipe["preprocessor"].transform(x)
predicted_labels = pipe["clusterer"]["kmeans"].labels_

# Evaluate the clustering performance using silhouette score and adjusted Rand index
silhouette = silhouette_score(preprocessed_data, predicted_labels)
adjusted_rand = adjusted_rand_score(y, predicted_labels)

print(f"Silhouette Score: {silhouette}")
print(f"Adjusted Rand Score: {adjusted_rand}")

# Create a dataframe with PCA components and predicted clusters
pca = pd.DataFrame(
    pipe["preprocessor"].transform(x),
    columns=["component_1", "component_2"],
)

print(pca["component_1"])

plt.scatter(pca["component_1"], pca["component_2"])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


pca["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pca["true_label"] = label_encoder.inverse_transform(y)

# Display the dataframe
print(pca)

# Plot the data points with different colors for each cluster and style for true labels
plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))
scat = sns.scatterplot(
    x=pca["component_1"],
    y=pca["component_2"],
    hue=pca["predicted_cluster"],
    style=pca["true_label"],
    palette="Set2"
)
scat.set_title("Clustering results from Steel Alloys Manufacturing process")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()

