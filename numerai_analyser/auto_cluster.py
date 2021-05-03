from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import umap
from apricot import FacilityLocationSelection

import joblib
# from dask.distributed import Client, progress

# client = Client(processes = False, threads_per_worker = 16, n_workers = 1)

# import multiprocessing as mp

# 1) Reduce dimensions - maybe add this later seems a little overkill
# 2) Cluster based on test data
# 3) Use previous cluster model to generate new assignments

class ClusterFeature:

    def __init__(self, data, clusters = None, min_clusters = 5, max_clusters = 6, reduce_dim = True, components = 4):

        # self.components = components
        self.clusters = clusters
        self.max_cluster = max_clusters
        self.min_cluster = min_clusters
        self.reduce_dim = reduce_dim
        self.components = components
        self.reducer = umap.UMAP(components)


        # if reduce_dim:
        #     print('Applying Apricot')
        #     self.reduceDimensionsModelFit(data)

        print('Estimating using kmeans')
        self.estimateCenters(data)


    def reduceDimensions(self, data):

        return pd.DataFrame(self.reducer.transform(data))


    def reduceDimensionsModelFit(self, data):

        n = data.shape[0]

        if n > 1000:

            sampler = FacilityLocationSelection(500, metric='euclidean', optimizer='lazy')

            data = sampler.fit_transform(data.to_numpy())

        self.reducer.fit(data)

    def range_d(self, start, end, step):
        output = []
        i = start

        while i > end:
            output.append(i)
            i = i - step

        return(output)



    def estimateCenters(self, data):

        # if self.reduce_dim:
        #     data = self.reduceDimensions(data)

        if self.clusters is None:

            # with joblib.parallel_backend('dask'):
            if True: # catch to undo parallelsation if needed for debugging

                self.models = [KMeans(n_clusters = i) for i in range(self.min_cluster, self.max_cluster + 1)]

                # pool = mp.Pool()
                self.models = list(map(lambda x: x.fit(data), self.models))

                n = len(self.models) - 1

                self.sse = list(map(lambda x: x.inertia_, self.models))
                line = list(self.range_d(self.sse[0], self.sse[n], (self.sse[0] - self.sse[n])/n))

                # print(self.sse)
                # print(line)
                dist_line = list(map(lambda x: x[1] - x[0], zip(self.sse, line)))
                # print(dist_line)

                edge = max(dist_line)
                self.best = [i for i, j in enumerate(dist_line) if j == edge]

                # print(self.best)

                self.cluster_model = self.models[self.best[0]]





        else:
            self.cluster_model = KMeans(n_clusters = self.clusters)

        self.cluster_model.fit(data)

    def assignClusters(self, data):

        # if self.reduce_dim:
        #     data = self.reduceDimensions(data)

        return self.cluster_model.predict(data)


if __name__ == "__main__":
    import random

    n = 1010

    test_data = pd.DataFrame({"one": [random.normalvariate(-i,1) for i in range(n)],
        "two": [random.normalvariate(i,1) for i in range(n)],
        "three": [random.normalvariate(i,1) for i in range(n)]})

    new_data = 	pd.DataFrame({"one": [random.normalvariate(-i,1) for i in range(n)],
        "two": [random.normalvariate(i,1) for i in range(n)],
        "three": [random.normalvariate(i,1) for i in range(n)]})

    print(test_data)

    cluster = ClusterFeature(test_data, None)

    print(cluster.assignClusters(new_data))
    print(np.unique(cluster.assignClusters(new_data)))


    print(cluster.range_d(5.1,3.5,0.2))





