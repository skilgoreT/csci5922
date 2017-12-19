#!/usr/bin/env python3

####################################################################
# FILE INFORMATION
####################################################################
# FILE: CLUSTER_ANALY.PY (CSCI 5622 PROJECT)
# DATE: 12/2/17
# AUTHORS: 
# FILE DESCRIPTION:
#    Perform automatic EDA on discovered clusters (use clusters.py
#      to discover clusters using kmodes). Goal is to gain 
#      some rough insights as to what each cluster actually means.
# ------------------------------------------------------------------

####################################################################
#  IMPORTS
####################################################################
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from pymongo import MongoClient
from math import isclose
from bson.objectid import ObjectId

####################################################################
#  VARIABLES AND CONSTANTS
####################################################################
SCRNSHOT_DOWNLD_CNT = 3 # number of screenshots to download per example

####################################################################
#  FUNCTION DEFINITIONS
####################################################################
class cluster_analyzer:
    def __init__(self, db_collection, outdir="./out/", mode="string", screenshots=False):
        self.outdir = os.path.normpath(outdir)
        self.db = db_collection
        self.svd = None
        self.tsne = None
        self.mode = mode
        self.screenshots = screenshots
        # create output directory if it doesn't exist
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def query_example_id(self, id):
        return self.db.find_one({'_id': id})

    def query_traits(self, qual_id):
        cursor = self.db.find({"qual_id": qual_id}, projection=['behavioral_traits'])
        traits = {}
        trait_set = set()
        for x in cursor:
            if len(x['behavioral_traits']) > 0:
                traits[str(x['_id'])]:x['behavioral_traits']
                #print(str(x['_id']))
                for t in x['behavioral_traits']:
                    trait_set.add(t)
        return trait_set, traits

    def query_ex_traits(self, xid):
        cursor = self.db.find_one({"_id":ObjectId(xid)}, projection=['behavioral_traits'])
        return cursor['behavioral_traits']

    # do EDA on the clusters found, save info to disk
    def analyze(self, X, Y, centroids, example_ids, qual_id, features):
        print("*** ANALYZE: Performing EDA on clusters ***")
        print("  - shape(X):{}, shape(Y):{}, # centroids:{}, len(ids):{}".
            format(X.shape, Y.shape, len(centroids), len(example_ids)))
        #print(example_ids)
        most_common = Counter(Y).most_common(8)
        most_common_list = [y for y,c in most_common]
        self.write_stats(X, Y, centroids, example_ids, qual_id, features)
        if self.mode is "num":
            self.plot_clusters(X, Y, centroids, 10)
        if self.screenshots:
            self.save_screenshots(X, Y, most_common_list, example_ids)

    def write_stats(self, X, Y, centroids, example_ids, qual_id, features):
        print("Writing cluster summary information...")
        # extract traits
        trait_set, traits = self.query_traits(qual_id)
        # first, write high-level summary.
        count_y = Counter(Y)
        most_common_list = [(y,c) for y,c in count_y.most_common(8)]
        ex = self.query_example_id(example_ids[0])
        path = os.path.normpath(self.outdir + '/' + 'SUMMARY.txt')
        f = open(path, 'w')
        f.write("Summary of clustering results for problem [qual_id:{}]\n".format(qual_id))
        f.write(" * Problem set: {}\n".format(ex['problem_set']))
        f.write(" * Problem text: '{}'\n".format(ex['txt']))
        f.write(" * Level description: {}\n".format(ex['level_summary']['description']))
        f.write("------------------------------------------------------\n")
        f.write(" * Number of examples clustered: {}\n".format(len(X)))
        f.write(" * Number of clusters: {}\n".format(len(centroids)))
        f.write(" * Most common clusters (label, count): {}\n".format(most_common_list))
        f.write("------------------------------------------------------\n")
        f.write("Behavioral Traits: {}\n".format(trait_set))
        f.write("------------------------------------------------------\n")
        f.write("Number of features extracted: {}\n".format(len(features)))
        f.write("All features extracted:\n")
        for i in range(len(features)):
            f.write("[{}]: {}\n".format(i, features[i]))
        f.write("------------------------------------------------------\n")
        f.write("All centroids:\n")
        f.write("  [LABEL_ID: CNT]\t [CENTROID]\n")
        label_counts = count_y.items()
        sorted_count = sorted(((k,v) for k,v in count_y.items()), key=lambda x: x[1], reverse=True)
        for k,v in sorted_count:
            f.write("  [{}: {}]\t{}\n".format(k, v, centroids[k]))
        f.close()
        # write cluster-level summaries
        for label,cnt in most_common_list:
            filt_X = [x for x,y in zip(X,Y) if y == label]
            filt_ids = [z for y,z in zip(Y,example_ids) if y == label]
            self.write_cluster_stats(label, filt_X, centroids[label], filt_ids, features, qual_id)

    def write_cluster_stats(self, cluster, X, centroid, example_ids, features, qual_id):
        path = self.get_cluster_output_dir_path(cluster)
        path = os.path.normpath(path + '/_' + str(cluster) + '_summary.txt')
        f = open(path, 'w')
        f.write("Cluster results for problem [qual_id:{}], cluster [{}]\n".format(qual_id, cluster))
        f.write("------------------------------------------------------\n")
        f.write(" - Number of examples in cluster {}: {}\n".format(cluster, len(X)))
        f.write("------------------------------------------------------\n")
        f.write("Centroid:\n")
        f.write(str(centroid) + '\n')
        f.write("------------------------------------------------------\n")
        f.write("Non-null features:\n")
        for i in range(len(centroid)):
            if self.mode is "string":
                if centroid[i] != "":
                    f.write("[feature {}]: {}\n".format(i, features[i]))
            else:
                if not isclose(centroid[i], -1, rel_tol=1e-5):
                    f.write("[feature {}]: {}\n".format(i, features[i]))
        f.write("------------------------------------------------------\n")
        f.write("Example feature vectors:\n")
        for i in range(min(5,len(X))):
            f.write("[{}]: {}\n".format(i, X[i]))
        f.write("------------------------------------------------------\n")
        f.write("ObjectIDs of examples in clusters + traits:\n")
        for i in range(len(example_ids)):
            traits = self.query_ex_traits(example_ids[i])
            f.write("  [{}]: {}\t{}\n".format(i, example_ids[i], traits))
        f.write("------------------------------------------------------\n")
        f.close()

    def get_cluster_output_dir_path(self, cluster):
        dirpath = self.outdir + '/' + str(cluster) + '/'
        dirpath = os.path.normpath(dirpath)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        return dirpath

    # download a few examples of screenshots from each of the most
    # common clusters.
    def save_screenshots(self, X, Y, filter_list, example_ids):
        print("Downloading screenshots...")
        for label in filter_list:
            # save each in own dir
            count = 0
            dirpath = self.get_cluster_output_dir_path(label)
            for y, obj_id in zip(Y, example_ids):
                if label == y:
                    # download screenshot url. only save first 5 for now
                    try:
                        filepath = dirpath + '/' + str(label) + '_' + str(count) + '.png'
                        filepath = os.path.normpath(filepath)
                        url = self.query_example_id(obj_id)['screenshot_url']
                        # seems amazon doesn't like python's user-agent
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        f = open(filepath, 'wb')
                        fweb = urllib.request.urlopen(req)
                        f.write(fweb.read())
                        f.close()
                        count += 1
                    except urllib.error.HTTPError:
                        print("Failed to download img {}".format(url))
                    if count >= SCRNSHOT_DOWNLD_CNT:
                        break

    # given a high-dimensional numeric feature vector, reduce to 
    # two dimensions for visualization
    def reduce_dim(self, X):
        if self.svd is None or self.tsne is None:
            # have to fit model the first time we use it
            self.svd = TruncatedSVD(n_components=40, random_state=0)
            self.tsne = TSNE(n_components=2, perplexity=40, verbose=2, learning_rate=400, n_iter=700)
            x_50 = self.svd.fit_transform(X)
            x_2 = self.tsne.fit_transform(x_50)
        else:
            # can't do this with TSNE
            # want to tranform data the same way every time
            #x_50 = self.svd.transform(X)
            #x_2 = self.tsne.transform(x_50)
            print("Error. tsne has no transform method.")
        return x_2

    # visualize using PCA-reduced input vector
    def plot_clusters(self, X, Y, centroids, n):
        print("Plotting clusters...")

        fig = plt.figure(1)
        # transform centroids alongside input vectors
        reduced_temp = self.reduce_dim(np.append(X, centroids, axis=0))
        reduced_cent = reduced_temp[-len(centroids):,:]
        reduced_X = reduced_temp[:len(X),:]

        # only plot top n centroid examples
        color_gen = (plt.cm.tab10(1.*i/n) for i in range(n))
        count_y = Counter(Y)
        most_common_list = [y for y,c in count_y.most_common(n)]
        for label in most_common_list:
            mask = (Y[:] == label)
            filt_X = reduced_X[mask,:]
            color = next(color_gen)
            plt.scatter(filt_X[:,0], filt_X[:,1], c=color, alpha=0.3, label=label)

        # plot points not part of a bigger cluster
        mask = [(Y[i] not in most_common_list) for i in range(len(Y))]
        mask = np.array(mask)
        filt_X = reduced_X[mask,:]
        plt.scatter(filt_X[:,0], filt_X[:,1], c='lightgray', alpha=0.5, label='other', marker='o')

        # add 'centroids' to plot for reference
        mask = [(i in most_common_list) for i in range(len(centroids))]
        mask = np.array(mask)
        filt_cent = reduced_cent[mask,:]
        plt.scatter(filt_cent[:,0], filt_cent[:,1], alpha=0.7, color='black', marker='+', label='centroids')
        # plot other centroids
        mask = np.logical_not(mask)
        filt_cent = reduced_cent[mask,:]
        plt.scatter(filt_cent[:,0], filt_cent[:,1], alpha=0.5, color='gray', marker='x', label='centroids')

        # TODO: plot CORRECT examples for reference
        plt.legend()
        path = os.path.normpath(self.outdir + '/' + 'fig_1.png')
        plt.savefig(path)
        plt.close(fig)


