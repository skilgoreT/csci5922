####################################################################
# FILE INFORMATION
####################################################################
# FILE: CLUSTER-ONE.PY (CSCI 5622 PROJECT)
# DATE: 12/3/17
# AUTHORS: 
# FILE DESCRIPTION:

# Generates clusters using the response fields for a given qual_id.
# Clustering algortihm k-modes is used. Clusters are generated where every
# feature of a record matches exactly to a given centroid. If no such centroid
# exists, then the record is assigned by default to the first centroid.
# If a document assigned to a centroid has behavioral traits, those traits will
# also be assigned to the centroid.

# These clusters are useful since every assigned document will share all
# response fields. Their answers will be the same and their canvases will look
# highly similar in a way that can categorized.

# New documents can then be cross-referenced with these generated centroids to
# compare assigned behavioral_traits.

# Clusters and centroids are used almost synonymously. A cluster is a grouping
# of records, whereas a centroid is what makes the records similar.
# Fit and train ared used synonymously.
# Predict and test are used synonymously.
# Generate clusters which represent document's response fields for a given
# qual_id
# Given the clusters, compare a document's response field with the generated
# clusters
# If a set of documents don't have responses when being fitted, then clusters
# won't be generated.
# If a document doesn't have a response when being predicted, a centroid will
# not be returned.

# It is often advantagous to transform the data before we do numeric
# (non-discrete) clustering, e.g. KMeans. Examples of tranformations
# are whitening, normalization, and dimensionality reduction. 
# For this reason, we will transform the extracted feature vector,
# but also keep a copy in order to do analysis of clusters later.
# This could also facilitate comparisons of clusters in the future.

####################################################################
#  IMPORTS
####################################################################
import argparse
from pymongo import MongoClient
import json
import numpy as np
# KModes can be found at https://github.com/nicodv/kmodes
from sklearn.cluster import KMeans
from collections import Counter
import csv
import sys
from cluster_analy import cluster_analyzer
from pprint import pprint

####################################################################
#  VARIABLES AND CONSTANTS
####################################################################
# db connection settings
client = MongoClient()
database = client.woot
collection = database.core
centroid_db = database.centroids

# program parameters
MISSING_FEATURE_VALUE = 0
NUM_CLUSTERS = 225
ANALYS_OUT_DIR = None

# behavior flags set by script arguments
FLAG_VERBOSE = False

####################################################################
#  CLASS / FUNCTION DEFINITIONS
####################################################################
class Featurizer:
    def __init__(self):
        self.features = {}
        self.num_features = 0
        return

    # fit featurizer to data. must call fit before transforming data.
    # extracts dictionary of features for later transformation.
    def fit(self, X_generator):
        if FLAG_VERBOSE:
            print("Compiling dictionary of features.")
        # Compile dictionary of all possible features in given list of records
        self.num_examples = 0
        features = {}
        for doc in X_generator:
            doc_features = {}
            doc_features = self.retrieveKeys(doc, doc_features)
            features = self.mergeFeatures(doc_features, features, "")
            self.num_examples += 1
        if FLAG_VERBOSE:
            print("Feature compilation complete.")
        self.num_features = self.countFeatures(features)
        self.features = features
        return self.num_features

    # transforms raw db entries into a numpy feature vector.
    # feature vector can be used as input into numeric clustering
    # algorithms.
    def transform(self, X_generator):
        X = np.zeros((self.num_examples, self.num_features))
        i = 0
        for doc in X_generator:
            X[i,:] = self.addFeatures(self.features, [], doc)
            i += 1
        if FLAG_VERBOSE:
            print("Finished transforming features")
            print(X)
            #print(repr(student_data))
        return X

    # return sequence of keys that features are built from
    # when extracted from the database
    def get_feature_names(self):
        return self.interpretFeatures(self.features, [])

    # Create dictionary of features for a passed record
    def retrieveKeys(self, response, features):
        for ex in response:
            if type(response[ex]) is dict:
                features[ex] = {}
                features[ex] = self.retrieveKeys(response[ex], features[ex])
            elif type(response[ex]) is list:
                continue
                #features[ex] = "X Dropped -- list."
            elif type(response[ex]) is str:
                continue
            else:
                features[ex] = str(type(response[ex]))
        return features

    # Adds features from records to global feature dictionary
    # record db's "key:type(value)" as the value for each feature key
    def mergeFeatures(self, doc_features, features, prefix):
        for ex in doc_features:
            if type(doc_features[ex]) is dict:
                if ex not in features:
                    features[ex] = {}
                features[ex] = self.mergeFeatures(doc_features[ex], features[ex], prefix + str(ex) + '.')
            elif type(doc_features[ex]) is list:
                continue
            #elif type(doc_features[ex]) is str:
            #    continue
            elif ex not in features:
                #features[ex] = doc_features[ex] # record type(response[ex])
                features[ex] = prefix + str(ex) + ':' + doc_features[ex] # record type(response[ex])
        return features

    # Add features to a passed record
    def addFeatures(self, features, student_data, response):
        for ex in features:
            if type(features[ex]) is dict:
                if ex not in response:
                    response[ex] = {}
                student_data = self.addFeatures(features[ex], student_data, response[ex])
            else:
                if ex not in response:
                    student_data.append(MISSING_FEATURE_VALUE)
                elif response[ex] is None:
                    student_data.append(MISSING_FEATURE_VALUE)
                else:
                    student_data.append(float(response[ex]))
        return student_data

    # Count number of features
    def countFeatures(self, features):
        length = 0
        for ex in features:
            if type(features[ex]) is dict:
                length += self.countFeatures(features[ex])
            else:
                length += 1
        return length


    # print in-order-list of feature keys with context, for 
    # interpreting feature vectors
    def interpretFeatures(self, features, output):
        for ex in features:
            if type(features[ex]) is dict:
                output = self.interpretFeatures(features[ex], output)
            else:
                output.append(features[ex])
        return output


####################################################################
#  Class BaseCluster: Cluster models inherit from this
####################################################################
class BaseClusterer:
    def __init__(self, n_clusters):
        return

    # Fit the model, should already have the feature vector and
    # provide it as an input 
    # param(in): X: Input Vector
    def fit(self, X):
        # do data transformation
        # fit the model (do clustering)
        # not implemented in base class
        pass

    # Generate feature vector from qual_id and number of documents
    # Note: More than 500 documents results in slow training
    def transform(self, qual_id, count):
        pass


####################################################################
#  Class KMeansClusterer: Do clustering using KMeans
####################################################################
class KMeansClusterer(BaseClusterer):
    def __init__(self, n_clusters):
        self.km = KMeans(n_clusters=n_clusters, n_init=8, verbose=False)

    # Fit the model, should already have the feature vector and
    # provide it as an input 
    # param(in): X: Input Vector
    def fit(self, X):
        # do data transformation
        # fit the model (do clustering)
        print("Clustering...")
        self.km.fit(X)
        print("Finished.")

    # Generate feature vector from qual_id and number of documents
    # Note: More than 500 documents results in slow training
    def transform(self, X):
        return X

    # Generate centroids given a qual_id and an amount of document to be trained
    #   Basically first calls transform() then calls fit()
    # Note: More than 500 documents results in slow training
    def fit_transform(self, X):
        X = self.transform(X)
        self.fit(X)

    # perform post-clustering analysis
    def analyze(self, qual_id, data, raw_X, X_ids, featurizer, analyzer=None):
        # Print important information from clustering
        # Centroids are common values to each cluster
        centroids = self.km.cluster_centers_
        if FLAG_VERBOSE:
            print("*** CENTROIDS: ***")
            print(centroids)

        # Labels is a list indicating which cluster each record belongs to
        labels = self.km.labels_
        if FLAG_VERBOSE:
            print("*** LABELS: ***")
            print(labels)

        # Cost is value indicating possible error in the clusters. Ideal value is 0.0
        if FLAG_VERBOSE:
            cost = self.km.inertia_
            print("*** COST: ***")
            print(cost)

        # Prints 5 largest cluster labels and number of records per cluster.
        if FLAG_VERBOSE:
            most_common = Counter(labels).most_common(5)
            print("Most populated centroids")
            print(most_common)

        # perform post-clustering analysis
        if analyzer:
            analyzer.analyze(raw_X, labels, centroids, X_ids, qual_id, 
                             featurizer.get_feature_names())


####################################################################
#  UTILITY FUNCTIONS
####################################################################
def query_db(qual_id, count):
    # Query for passed qual_id with incorrect answers
    # May take a lengthly amount of time. Recommend optimizing query.
    print("Querying for", qual_id)
    curser = collection.find({"qual_id": qual_id, "correct": False})[:count]
    if FLAG_VERBOSE:
        print("Query complete.")
    return curser

# provide iterator to extract features from db documents
def exampleGenerator(db_curser):
    for doc in db_curser:
        if doc['response'] is None:
            continue
        else:
            yield doc['response']

# get example IDs
def query_example_ids(db_curser):
    db_curser = db_curser.rewind()
    X_ids = []
    for doc in db_curser:
        if doc['response'] is None:
            continue
        else:
            X_ids.append(doc['_id'])
    return X_ids

####################################################################
#  MAIN
####################################################################
# Run KMEANS on a subset of examples from a given Qual ID
def main():
    desc = "Run clustering on woot math problems (per-problem clustering). This script expects a QualID, and will perform clustering just on that one problem."
    epil = "If you choose to run additional analysis, this script will print summaries of the most common clusters and download some example screenshots."
    argparser = argparse.ArgumentParser(description=desc, epilog=epil)
    argparser.add_argument('qual_id', help="Qual ID of problem to analyze")
    argparser.add_argument('-k', '--num_clusters', help="number of clusters to find (default:225)",
                           required=False, default=225, type=int)
    argparser.add_argument('-n', '--num_ex', help="Number of examples of qual_id to use (default:500)",
                           required=False, default=500, type=int)
    argparser.add_argument('-o', '--output_dir', help="output directory for analysis. Default: './out/<qual_id>/'",
                           required=False)
    argparser.add_argument('-s', '--save_screenshots', help="Save screenshots?",
                           action='store_true', default=False, required=False)
    argparser.add_argument('-v', '--verbose', help="Print debug info",
                           action='store_true', required=False)
    args = argparser.parse_args()

    # fill out script parameters
    if args.num_clusters is not None:
        NUM_CLUSTERS = args.num_clusters
    ANALYS_OUT_DIR = args.output_dir
    FLAG_VERBOSE = args.verbose
    qual_id = args.qual_id

    # Generate feature vector from qual_id and number of documents
    # Note: More than 500 documents results in slow training
    # Find all numeric features and transform data into a feature vector
    db_curser = query_db(qual_id, args.num_ex)
    featurizer = Featurizer()
    num_features = featurizer.fit(exampleGenerator(db_curser))
    print("*** Number of features: {}".format(num_features))
    if num_features == 0:
        return
    # Reuse queried documents, extract features from them.
    db_curser = db_curser.rewind()
    student_data = featurizer.transform(exampleGenerator(db_curser))
    num_examples = student_data.shape[0]
    print("*** Number of non-empty records for [Q_ID:{}]: {}".format(qual_id, num_examples))
    # set up Cluster Analyzer to to EDA on clusters found
    data_ids = query_example_ids(db_curser)
    out_dir = ANALYS_OUT_DIR
    if out_dir is None:
        out_dir = "./out/" + str(qual_id) + '/'
    analys = cluster_analyzer(collection, out_dir, mode="num", screenshots=args.save_screenshots)

    # run a couple of models, and compare clusters
    # KMEANS
    model = KMeansClusterer(NUM_CLUSTERS)
    model.fit_transform(student_data)
    model.analyze(qual_id, db_curser, student_data, data_ids, featurizer, analys)

    # DBSCAN
    #model = 

if __name__ == '__main__':
    main()

