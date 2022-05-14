#!/usr/bin/env python3

import sys
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.cluster import OPTICS, DBSCAN

DATASETS = {
    0: 'datasets/small_test.csv',
    1: 'datasets/MopsiLocationsUntil2012-Finland.csv',
    2: 'datasets/MopsiLocations2012-Joensuu.csv',
    3: 'datasets/BPD_Part_1_Victim_Based_Crime_Data.csv'
}

def main() -> None:
    print('Available dataset for OPTICS algorithm:')
    print('    0 - Small test set')
    print('    1 - Mopsi Locations Finland')
    print('    2 - Mopsi Locations Joensuu')
    print('    3 - Crime in Baltimore')

    # Choose a dataset
    dataset = int(input('Please enter a value for the dataset: '))
    if 3 < dataset < 1:
        print('Please choose a valid dataset!')
        sys.exit()

    header = ['Latitude', 'Longitude']

    # Read a dataset
    with open(DATASETS[dataset], 'r') as csv_file:
        tmp_data = pd.read_csv(csv_file, delimiter=',', usecols=header, encoding='utf-8')

    data = np.array(tmp_data, np.double)
    del tmp_data

    if dataset == 1:
        data /= 10000
    elif dataset == 3:
        data = data[~np.isnan(data).all(axis=1), :]

    print('The number of samples in the dataset is: ' + str(len(data)))
    print('First three entries in the dataset are:')
    print(data[:3])

    # Read the parameters
    min_pts = \
        int(input('Please choose the minimum number of points in a cluster: '))

    # Fit cluster
    clust = OPTICS(min_samples=min_pts)
    clust.fit(data)

    # Compute some statistics
    labels = clust.labels_[clust.ordering_]
    unique_labels_with_outliers = len(set(labels))
    unique_labels_count = unique_labels_with_outliers - 1 if -1 in labels else unique_labels_with_outliers
    negative_labels_count = len(data[clust.labels_ == -1])

    print('The number of clusters in the dataset is: ' + str(unique_labels_count))
    print('The number of outliers in the dataset is: ' + str(negative_labels_count))

    if unique_labels_count == 0:
        return

    for label in range(0, unique_labels_count):
        cluster_points = data[clust.labels_ == label]
        print(f'Cluster {label} has {len(cluster_points)} samples.')

    # Plot the results
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, :])
    colors = ['g.', 'r.', 'b.', 'y.', 'c.', 'm.']

    # Plot the clusters
    for cluster, color in zip(range(0, unique_labels_count), colors):
        cluster_points = data[clust.labels_ == cluster]
        ax1.plot(cluster_points[:, 0], cluster_points[:, 1], color, alpha=0.3)
    ax1.plot(data[clust.labels_ == -1, 0], data[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax1.set_title('OPTICS Clustering')

    # Plot the reachability graphic
    space = np.arange(len(data))
    reachability = clust.reachability_[clust.ordering_]

    for cluster, color in zip(range(0, unique_labels_count), colors):
        Xk = space[labels == cluster]
        Rk = reachability[labels == cluster]
        ax2.plot(Xk, Rk, color, alpha=0.3)
    ax2.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax2.set_ylabel('Reachability (epsilon)')
    ax2.set_title('Reachability Plot')

    plt.tight_layout()
    plt.show()

    # Evaluation of the results
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, clust.labels_)
    print('The Calinski-Harabasz score is: ' + str(calinski_harabasz_score))
    davies_bouldin_score = metrics.davies_bouldin_score(data, clust.labels_)
    print('The Davies–Bouldin score is: ' + str(davies_bouldin_score))
    silhouette_score = metrics.silhouette_score(data, clust.labels_, metric='euclidean')
    print('The Silhouette score is: ' + str(silhouette_score))

    # Compare with DBSCAN

    # Fit cluster
    clust = DBSCAN(min_samples=min_pts)
    clust.fit(data)

    # Compute some statistics
    labels = clust.labels_
    unique_labels_with_outliers = len(set(labels))
    unique_labels_count = unique_labels_with_outliers - 1 if -1 in labels else unique_labels_with_outliers
    negative_labels_count = len(data[clust.labels_ == -1])

    print('The number of clusters in the dataset is: ' + str(unique_labels_count))
    print('The number of outliers in the dataset is: ' + str(negative_labels_count))

    if unique_labels_count == 0:
        return

    for label in range(0, unique_labels_count):
        cluster_points = data[clust.labels_ == label]
        print(f'Cluster {label} has {len(cluster_points)} samples.')

    # Plot the results
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(G[0, 0])
    colors = ['g.', 'r.', 'b.', 'y.', 'c.', 'm.']

    # Plot the clusters
    for cluster, color in zip(range(0, unique_labels_count), colors):
        cluster_points = data[clust.labels_ == cluster]
        ax1.plot(cluster_points[:, 0], cluster_points[:, 1], color, alpha=0.3)
    ax1.plot(data[clust.labels_ == -1, 0], data[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax1.set_title('DBSCAN Clustering')

    plt.tight_layout()
    plt.show()

    # Evaluation of the results
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, clust.labels_)
    print('The Calinski-Harabasz score is: ' + str(calinski_harabasz_score))
    davies_bouldin_score = metrics.davies_bouldin_score(data, clust.labels_)
    print('The Davies–Bouldin score is: ' + str(davies_bouldin_score))
    silhouette_score = metrics.silhouette_score(data, clust.labels_, metric='euclidean')
    print('The Silhouette score is: ' + str(silhouette_score))

if __name__ == '__main__':
    main()
