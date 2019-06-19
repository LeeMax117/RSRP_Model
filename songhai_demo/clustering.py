import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from itertools import cycle, islice


# prepare data
df = pd.read_csv('train.csv')
offline_location, offline_RSRP = np.array(df.iloc[:, [0, 1]]), np.array(df.iloc[:, 2:])

# set parameters
plt_num = 1
params = {'eps': 20,
          'n_clusters': 7,
          'min_cluster_size': 0.1}

# model configuration
two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity='nearest_neighbors')
dbscan = cluster.DBSCAN(eps=params['eps'])

clusterting_algorithms = (
    ('MiniBatchKMeans', two_means),
    ('SpectralClustering', spectral),
    ('DBSCAN', dbscan)
)

# solving model
for name, algorithm in clusterting_algorithms:
    t0 = time.time()

    algorithm.fit(offline_RSRP)

    t1 = time.time()
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(offline_RSRP)

    plt.subplot(1, len(clusterting_algorithms), plt_num)
    plt.title(name, size=12)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00',
                                         '#8B7500', '#87CEFF', '#B7B7B7',
                                         '#CD3333', '#CD5B45', '#C0FF3E']), int(max(y_pred) + 1))))
    colors = np.append(colors, ["#000000"])
    plt.scatter(offline_location[:, 0], offline_location[:, 1], s=10, c=colors[y_pred])

    plt.text(.99, .01, ('t: %.2fs, c: %d' % ((t1 - t0), int(max(y_pred)))).lstrip('0'),
             transform=plt.gca().transAxes, size=10,
             horizontalalignment='right')
    plt_num += 1
plt.show()