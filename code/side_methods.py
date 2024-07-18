import numpy as np
import copy

import CFTree_methods # local file

#
# Distances
#

def average_intercluster_distance_D2(CF_1, CF_2): #TODO: verify that it is correct
    num = CF_2.N * CF_1.SS + CF_1.N * CF_2.SS - 2 * np.sum(CF_1.LS * CF_2.LS)
    D2 = np.sqrt(num / (CF_1.N * CF_2.N))
    return D2


def centroid_euclidean_distance_D0(centroid_1, centroid_2):
    pow_2 = np.sum((centroid_1 - centroid_2) ** 2)
    return pow_2 ** (1/2)


#
# For phase 3 and phase 4
#

def hierarchical_clustering(list_subclusters, num_clusters, distance = average_intercluster_distance_D2):
    num_initial_clusters = len(list_subclusters)

    # Initialize clusters with their indices
    clusters = {i: [i] for i in range(num_initial_clusters)}

    # Compute pairwise distance matrix
    distance_matrix = np.zeros((num_initial_clusters, num_initial_clusters))

    for i in range(num_initial_clusters):
        for j in range(i + 1, num_initial_clusters):
            distance_matrix[i, j] = distance(list_subclusters[i][0], list_subclusters[j][0])
            distance_matrix[j, i] = distance_matrix[i, j]

    # Agglomerative clustering process
    while len(clusters) > num_clusters:
        # Find the pair of clusters with the smallest distance
        min_dist = np.inf
        min_pair = None
        for i in clusters:
            for j in clusters:
                if i < j:
                    dist = distance_matrix[i, j]
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = [i, j]

        # Merge the closest pair of clusters
        merge_i = min_pair[0]
        merge_j = min_pair[1]

        new_cluster_indices = clusters[merge_i] + clusters[merge_j]

        distance_matrix[merge_j, :] = np.inf
        distance_matrix[:, merge_j] = np.inf
        
        # Update distance matrix
        new_CF = CFTree_methods.CF()
        new_CF.merge_CF(list_subclusters[merge_i][0])
        new_CF.merge_CF(list_subclusters[merge_j][0])
        for j in range(len(list_subclusters)):
            if distance_matrix[merge_i, j] != np.inf:
                distance_matrix[merge_i, j] = distance(new_CF, list_subclusters[j][0])
                distance_matrix[j, merge_i] = distance_matrix[merge_i, j]            

        # Remove old clusters from the cluster list and add merged cluster
        del clusters[merge_j] # no holes since i < j
        clusters[merge_i] = new_cluster_indices


    # Assign final cluster labels and build final_clusters dictionary
    final_clusters = {}
    for orig_cluster_idx, cluster_indices in clusters.items():
        for idx in cluster_indices:
            final_clusters[idx] = orig_cluster_idx


    result_clusters = {}
    for subcluster_idx, cluster in final_clusters.items():
        # Initialize list if it doesn't exist for the current cluster
        if cluster not in result_clusters:
            result_clusters[cluster] = []

        # Iterate over datapoints and append to the appropriate cluster
        for datapoint_i in range(len(list_subclusters[subcluster_idx][1].datapoints)):
            result_clusters[cluster].append(list_subclusters[subcluster_idx][1].datapoints[datapoint_i])

    # Convert lists to NumPy arrays if needed
    for cluster, datapoints in result_clusters.items():
        result_clusters[cluster] = np.array(datapoints)
    
    # Rename the index of the clusters (to have indeces from 0 to n-1 where n is the number of clusters)
    old_keys = list(result_clusters.keys())
    for i, old_key in enumerate(old_keys):
        result_clusters[i] = result_clusters.pop(old_key)

    return result_clusters



def compute_centroid_X0(array_datapoints):
    num = np.sum(array_datapoints, axis=0)
    # TODO IMPROVEMENT: an alternative could be to compute X0 starting from the clustering feature of the cluster
    return num / len(array_datapoints)



def redistribute_datapoints(result_clusters, num_iterations):
    # TODO IMPROVEMENT: a further improvement could be to handle the outliers:
    #   if a datapoint is too far away from all the centroids, label it as outlier
    new_result_clusters = copy.deepcopy(result_clusters)
    for i in range(num_iterations):
        centroids = []
        for cluster_i in range(len(new_result_clusters)):
            centroids.append(compute_centroid_X0(new_result_clusters[cluster_i]))

        tmp_new_result_clusters = {i: [] for i in range(len(new_result_clusters))}
        for cluster_i in range(len(new_result_clusters)):
            for datapoint_i in range(len(new_result_clusters[cluster_i])):
                dist = np.inf
                current_choice_cluster = -1
                for centroid_i in range(len(centroids)):
                    tmp_dist = centroid_euclidean_distance_D0(centroids[centroid_i], new_result_clusters[cluster_i][datapoint_i])
                    if tmp_dist < dist:
                        dist = tmp_dist
                        current_choice_cluster = centroid_i
                tmp_new_result_clusters[current_choice_cluster].append(new_result_clusters[cluster_i][datapoint_i])
        
        # Convert lists back to numpy arrays
        for key in tmp_new_result_clusters:
            tmp_new_result_clusters[key] = np.array(tmp_new_result_clusters[key])

        new_result_clusters = tmp_new_result_clusters

    return new_result_clusters
