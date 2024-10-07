# -*- coding: utf-8 -*-
"""
Ana P. O. Costa et. al. "Manufacturing process encoding through natural language processing for prediction of material properties"
2023
"""

# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Read the dataset from a CSV file and select relevant columns
steel_df = pd.read_csv('steel_database.csv', sep=';', low_memory=False)
X = pd.DataFrame(steel_df, columns=['Cr', 'Al', 'B', 'Co', 'Mo', 'Ni', 'Ti', 'Zr', 'C', 'Fe', 'Pd',
                                    'Mn',	'P',	'Si', 'N',	'Cu',	'Nb',	'Se',	'Ta',	'W',	'V',	'S',
                                   'e1',	'e2',	'e3',	'e4','e5', 'e6', 'e7',	'e8',	'e9',	'e10', 'e11', 'e12','e13','e14', 'e15',
                                         'e16',	'e17',	'e18',	'e19', 'e20', 'e21','e22','e23',
                                         'e24',	'e25',	'e26',	'e27',	'e28', 'e29','e30','e31',
                                         'e32', 'e33',	'e34',	'e35',	'e36',	'e37', 'e38', 'e39',
                                         'e40', 'e41', 'e42',	'e43',	'e44',	'e45',	'e46', 'e47',
                                         'e48','e49','e50', 'e51',	'e52',	'e53',	'e54',	'e55', 'e56',
                                         'e57','e58','e59', 'e60',	'e61',	'e62',	'e63',	'e64', 'e65',
                                         'e66','e67','e68', 'e69',	'e70',	'e71',	'e72',	'e73', 'e74',
                                         'e75','e76','e77', 'e78',	'e79',	'e80',	'e81',	'e82', 'e83',
                                          'e84','e85','e86', 'e87',	'e88',	'e89',	'e90',	'e91', 'e92',
                                           'e93','e94','e95', 'e96',	'e97',	'e98',	'e99',	'e100', 'e101',
                                          'e102','e103','e10', 'e105', 'e106','e107','e108', 'e109', 'e110',
                                         'e111','e112','e113', 'e114','e115',	'e116',	'e117',	'e118',	'e119',
                                         'e120', 'e121','e122', 'e123', 'e124', 'e125', 'e126', 'e127',
                                        'e128', 'e129', 'e130', 'e131','e132', 'e133', 'e134', 'e135','e136', 'e136', 'e137'
                                        'e135',	'e136',	'e137',	'e138',	'e139',	'e140',	'e141',	'e142',	'e143',	'e144',
                                        'e145',	'e146',	'e147',	'e148',	'e149',	'e150',	'e151',	'e152',	'e153',	'e154',
                                        'e155',	'e156',	'e157',	'e158',	'e159',	'e160',	'e161',	'e162',	'e163',	'e164',
                                        'e165',	'e166',	'e167',	'e168',	'e169',	'e170',	'e171',	'e172',	'e173',	'e174',
                                        'e175',	'e176',	'e177',	'e178',	'e179',	'e180',	'e181',	'e182',	'e183',	'e184',
                                        'e185',	'e186',	'e187',	'e188',	'e189',	'e190',	'e191',	'e192',	'e193',	'e194',
                                        'e195',	'e196',	'e197',	'e198',	'e199',	'e200',	'e201',	'e202',	'e203',	'e204',
                                        'UTS', 'HB', 'YS', 'Elongation'])

# Convert data to float32 and replace NaN values with 0
X = X.astype(np.float32)
X = X.replace(np.nan, 0)

# Save the preprocessed dataframe to a new CSV file
X.to_csv('X.csv', sep=';', index=False)

# Convert the dataframe to a numpy array
X = X.values

# Apply PCA to reduce data to 2 dimensions
preprocessor = Pipeline([("scaler", MinMaxScaler()), ("pca", PCA(n_components=2, random_state=84))])
pipe = Pipeline([("preprocessor", preprocessor)])
pipe.fit(X)
pcadf = pd.DataFrame(pipe["preprocessor"].transform(X), columns=["component_1", "component_2"])

# Create a k-means clustering object and fit it to the reduced data
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pcadf)

# Plot the 2D clustering results
plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))
plt.scatter(pcadf["component_1"], pcadf["component_2"], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, c='r')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('K-means Clustering with PCA')
plt.show()

# Apply PCA again, this time for 3 dimensions
preprocessor = Pipeline([("scaler", MinMaxScaler()), ("pca", PCA(n_components=3, random_state=42))])
pipe = Pipeline([("preprocessor", preprocessor)])
pipe.fit(X)
pcadf = pd.DataFrame(pipe["preprocessor"].transform(X), columns=["component_1", "component_2", "component_3"])

# Create a new k-means clustering object and fit it to the 3D reduced data
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pcadf)

# Plot the 3D clustering results
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')
ax.scatter(pcadf["component_1"], pcadf["component_2"], pcadf["component_3"], c=kmeans.labels_)
ax.set_xlabel('Component_1', fontsize=12, labelpad=10)
ax.set_ylabel('Component_2', fontsize=12, labelpad=10)
ax.set_zlabel('Component_3', fontsize=12, labelpad=10)
plt.show()