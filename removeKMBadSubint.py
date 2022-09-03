#!/usr/bin/env python
import sys,os,glob
import numpy as np
import subprocess
"""
remove bad archives due to snr and all:avg(flux) stat analysis
"""
# use K-Mmeans method to remove bad archives due to telescope error

def euclidean_distance(vecA, vecB):
    """ Get euclidean distance of two vectors """
    # return np.sqrt(np.sum(np.square(vecA - vecB)))
    return np.linalg.norm(vecA - vecB)

def random_centroids(data, k):
    """ K-Mmeans algorithm """
    dim = np.shape(data)[1]  # get the dimension of the data
    centroids = np.mat(np.zeros((k, dim)))
    for j in range(dim):  # Generate random numbers between the min and max in each dimension  
        min_j = np.min(data[:, j])
        range_j = np.max(data[:, j]) - min_j
        centroids[:, j] = min_j * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * range_j
    return centroids

def nearest(data, cluster_centers, distance_func=euclidean_distance):
    min_dist = np.inf
    m = np.shape(cluster_centers)[0]  # the number of clustering center initiated now 
    for i in range(m):
        d = distance_func(data, cluster_centers[i, ])  # cal the distance of each point with every clustering centers 
        if min_dist > d:  # choose the closest distance
            min_dist = d
    return min_dist

def get_centroids(data, k, distance_func=euclidean_distance):
    """ K-Mmeans++ algorithm """
    m, n = np.shape(data)
    cluster_centers = np.mat(np.zeros((k, n)))
    index = np.random.randint(0, m)  # 1. select a data point randomly as the first clustering cneter
    cluster_centers[0, ] = np.copy(data[index, ])
    d = [0.0 for _ in range(m)]  # 2. initializing a series of distances 
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            d[j] = nearest(data[j, ], cluster_centers[0:i, ], distance_func)  # 3. find the closest clustering point for very data point
            sum_all += d[j]  # 4. add all the closest distances
        sum_all *= np.random.random()  # 5. get random number between 0 & sum_all
        for j, di in enumerate(d):  # 6. set farstest data point as the clustering centers
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(data[j, ])
            break
    return cluster_centers 

def KMeans(data, k, distance_func=euclidean_distance):
    '''get clustering centers based on K-means algoriths'''
    m = np.shape(data)[0]  # get the row number m 
    cluster_assment = np.mat(np.zeros((m, 2)))  # Initiate a matrix, to record clustering indexes and store and distance^2
    # centroids = random_centroids(data, k)  # generate initiation points 
    centroids = get_centroids(data, k)
    cluster_changed = True  # determine if the clustering point needs to be recalculated 
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            distance_min = np.inf  # set the smallest distance of sample points and clustering centers, default value is inf 
            index_min = -1  #  category belonging to
            for j in range(k):
                distance_ji = distance_func(centroids[j, :], data[i, :])
                if distance_ji < distance_min:
                    distance_min = distance_ji
                    index_min = j
            if cluster_assment[i, 0] != index_min:
                cluster_changed = True
                cluster_assment[i, :] = index_min, distance_min ** 2  # store ditance^2
        for cent in range(k):  # update centroid, taking the mean of all the points in each clustering as the centroid 
            pts_in_cluster = data[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(pts_in_cluster, axis=0)
    return centroids, cluster_assment

def CreateandCheckSNRandFluxstats(fi):
    import matplotlib.pyplot as plt
    colors = ['C0', 'C1', 'C2']

    fluxfile = fi.split(".")[0]+'_flux.txt'
    clusterfile = fi.split(".")[0]+'_cluster'
    snrstatfig =  fi.split(".")[0]+'_cluster.png'
    print("Produce flux stats file")
    cmdline = "psrstat -jpF -l subint=0- -c snr -c length -c all:avg  %s > %s"%(fi, fluxfile)
    print("%s"%cmdline)
    subprocess.call(cmdline, shell=True)
    if 'Apple' in sys.version:
        #Macbook
        cmdline = "sed -i '' 's/=/\ /g' %s"%(fluxfile)
    else:
        #Linux
        cmdline = "sed -i 's/=/\ /g' %s"%(fluxfile)
    subprocess.call(cmdline, shell=True)

    avgflux = np.loadtxt(fluxfile, usecols=8)
    Nsubint = avgflux.size
    clusterdata = []
    fig = plt.figure(figsize=[8,4])

    # clustering snrs: 2clusters
    k = 2
    # flux analysis
    centroids, cluster_assment = KMeans(avgflux.reshape(-1,1), k)
    clusterdata.append(centroids)
    clusterdata.append(cluster_assment)
    # plot
    ax = fig.add_subplot(121)
    for idx in range(Nsubint):
        ax.scatter(idx, avgflux[idx], c=colors[int(cluster_assment[idx,0])], marker='x')
        ax.set_ylabel("Average flux")

    # clustering snrs: 2clusters
    k = 3
    # flux analysis
    centroids, cluster_assment = KMeans(avgflux.reshape(-1,1), k)
    clusterdata.append(centroids)
    clusterdata.append(cluster_assment)
    # plot
    ax = fig.add_subplot(122)
    for idx in range(Nsubint):
        ax.scatter(idx, avgflux[idx], c=colors[int(cluster_assment[idx,0])], marker='x')
        ax.set_ylabel("Average flux")

    fig.savefig(snrstatfig)
    plt.close()
    print("snrstat fig saved to %s"%(snrstatfig))
    np.save(fi.split(".")[0]+'_cluster', np.array(clusterdata))

def LabelBadArchs(fi, k_flux=None):
    clusterdata = np.load(fi.split(".")[0]+'_cluster.npy')
    if k_flux==2:
        centroids, cluster_assment = clusterdata[0], clusterdata[1]
    elif k_flux==3:
        centroids, cluster_assment = clusterdata[2], clusterdata[3]
    else:
        print("Only 2 & 3 are allowed, retry")
        exit(0)
    removeIdxes = np.where(cluster_assment[:,0]==np.argmax(centroids))[0]
    subints_remove = ' '.join((str(x) for x in removeIdxes))
    cmdline = "paz -w '%s' -m %s"%(subints_remove, fi)
    print(cmdline)

def _get_parser():
    """
    Arguments parser.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Use K-Mmeans algorithm clustering method to remove bad archives \
                    due to telescope error (removeBadArchs.py) @Heng Xu.20200722")
    parser.add_argument('files', nargs='+', help='Archive(s) to be look into' )
    parser.add_argument("-i", default=None, type=int, help="Obs indexes those need to be labeled.")
    parser.add_argument("-k_snr", default=None, type=int, help="Number of clusterings based on snrs, 2 and 3.")
    parser.add_argument("-k_flux",default=None, type=int, help="Number of clusterings based on flux, 2 and 3.")
    parser.add_argument("-label", action='store_true', help="Label bad subints")
    return parser.parse_args()

if __name__ == "__main__":
    import argparse,subprocess
    args = _get_parser()
    if args.label and args.k_flux:
        print("Labels bad subints using existing data...")
        for fi in args.files:
            LabelBadArchs(fi, args.k_flux)
    else:
        print("Analysis flux data and make diagnose plots...")
        for fi in args.files:
            CreateandCheckSNRandFluxstats(fi)
