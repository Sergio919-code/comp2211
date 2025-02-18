import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3 , n_init=20 , max_iter=300 , random_state=42)
df = pd.read_csv('./datas/K-means_data.csv')
df['Cluster'] = kmeans.fit_predict(df[['Feature1','Feature2','Feature3']])

fig = plt.figure()
ax = fig.add_subplot(111 , projection='3d')

ax.scatter(df['Feature1'] , df['Feature2'] , df['Feature3'] , c=df['Cluster'] , cmap='viridis' , s=20 , alpha=0.3)

centroids = kmeans.cluster_centers_
print(centroids)
ax.scatter(centroids[: , 0] , centroids[: ,1] , centroids[: , 2] , c='red' , s=100 , alpha=1 , label='Centroids')

ax.set_title("K-Means")
ax.set_xlabel('Feature1')
ax.set_ylabel('Feature2')
ax.set_zlabel('Feature3')
ax.legend()

plt.show()



