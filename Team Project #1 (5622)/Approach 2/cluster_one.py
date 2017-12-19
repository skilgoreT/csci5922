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

####################################################################
#  IMPORTS
####################################################################
import argparse
from pymongo import MongoClient
import json
import numpy as np
# KModes can be found at https://github.com/nicodv/kmodes
from kmodes.kmodes import KModes
from collections import Counter
import csv
import sys
from cluster_analy import cluster_analyzer
from pprint import pprint

####################################################################
#  VARIABLES AND CONSTANTS
####################################################################
client = MongoClient()
database = client.b
collection = database.anon_student_task_responses
centroid_db = database.centroids
NUM_CLUSTERS = 225
ANALYS_OUT_DIR = None

# behavior flags set by script arguments
FLAG_USE_CENTROID_DB = False
FLAG_VERBOSE = False
FLAG_DO_ANALYSIS = True

####################################################################
#  FUNCTION DEFINITIONS
####################################################################

# Create dictionary of features for a passed record
def retrieveKeys(response, features):
    for ex in response:
        if type(response[ex]) is dict:
            features[ex] = {}
            features[ex] = retrieveKeys(response[ex], features[ex])
        elif type(response[ex]) is list:
            continue
            #features[ex] = "X Dropped -- list."
        else:
            features[ex] = str(type(response[ex]))
    return features


# Adds features from records to global feature dictionary
# record db's "key:type(value)" as the value for each feature key
def mergeFeatures(doc_features, features, prefix):
    for ex in doc_features:
        if type(doc_features[ex]) is dict:
            if ex not in features:
                features[ex] = {}
            features[ex] = mergeFeatures(doc_features[ex], features[ex], prefix + str(ex) + '.')
        elif type(doc_features[ex]) is list:
            continue
        elif ex not in features:
            #features[ex] = doc_features[ex] # record type(response[ex])
            features[ex] = prefix + str(ex) + ':' + doc_features[ex] # record type(response[ex])
    return features


# Add features to a passed record
def addFeatures(features, student_data, response):
    for ex in features:
        if type(features[ex]) is dict:
            if ex not in response:
                response[ex] = {}
            student_data = addFeatures(features[ex], student_data, response[ex])
        else:
            #if features[ex][0] is 'X':
            #    continue
            if ex not in response:
                student_data = np.append(student_data, '')
            elif response[ex] is None:
                student_data = np.append(student_data, '')
            else:
                student_data = np.append(student_data, response[ex])
    return student_data


# Count number of features
def countFeatures(features):
    length = 0
    for ex in features:
        if type(features[ex]) is dict:
            length += countFeatures(features[ex])
        else:
            length += 1
    return length


# print in-order-list of feature keys with context, for
# interpreting feature vectors
def interpretFeatures(features, output):
    for ex in features:
        if type(features[ex]) is dict:
            output = interpretFeatures(features[ex], output)
        else:
            output.append(features[ex])
    return output


# Generate centroids given a qual_id and an amount of document to be trained
def fit(qual_id, count):
    # More than 500 documents results in slow training
    if count >= 500:
        count = 500
    # Query for passed qual_id with incorrect answers
    # May take a lengthly amount of time. Recommend optimizing query.
    if FLAG_VERBOSE:
        print("Querying for", qual_id)
    data = collection.find({"qual_id": qual_id, "correct": False})[:count]
    if FLAG_VERBOSE:
        print("Query complete.")

    # Compile dictionary of all possible features in given list of records
    if FLAG_VERBOSE:
        print("Compiling dictionary of features.")
    num_examples = 0
    num_empty = 0
    features = {}
    for doc in data:
        doc_features = {}
        if doc['response'] is None:
            num_empty += 1
            continue
        doc_features = retrieveKeys(doc['response'], doc_features)
        features = mergeFeatures(doc_features, features, "")
        num_examples += 1
    if FLAG_VERBOSE:
        print("Feature compilation complete.")

    # Count number of features
    num_features = countFeatures(features)
    if FLAG_VERBOSE:
        print("*** Number of features: {}".format(num_features))
        print("*** Number of non-empty records for [Q_ID:{}]: {}. (dropped {} with empty resp)".format(qual_id, num_examples, num_empty))
    if num_features == 0:
        return

    # Reuse queried documents.
    data = data.rewind()

    # Append missing features to all records and assign common benign value.
    # Current benign value is an empty string.
    # print("Appending features to documents.")
    # faster to create zeroed np array first, rather then appending
    student_data = np.zeros((num_examples, num_features), dtype='<U32')
    i = 0
    for doc in data:
        if doc['response'] is None:
            continue
        else:
            temp = addFeatures(features, [], doc['response'])
            student_data[i,:] = temp
            i += 1
    if FLAG_VERBOSE:
        print("Finished appending features to documents.")
        print(student_data)
    #print("*** Features: ***")
    #pprint(interpretFeatures(features, []))

    # print feature vectors
    #print("*** FEATURE VECTOR: ***")
    #i = 0
    #for row in student_data:
    #    print("[{}]: {}".format(i, row))
    #    i += 1
    #print(repr(student_data))

    # Perform k-modes clustering
    print("Clustering...")
    clusters = NUM_CLUSTERS
    # K-modes implementation can't generate more than 255 centroids
    if clusters > 255:
        clusters = 255
    if clusters > len(student_data):
        clusters = len(student_data)
    km = KModes(n_clusters=clusters, init='Cao', n_init=4, verbose=False)
    km.fit(student_data)
    print("Finished.")

    # Print important information from clustering
    # Centroids are common values to each cluster
    centroids = km.cluster_centroids_
    if FLAG_VERBOSE:
        print("*** CENTROIDS: ***")
        print(centroids)

    # Labels is a list indicating which cluster each record belongs to
    labels = km.labels_
    if FLAG_VERBOSE:
        print("*** LABELS: ***")
        print(labels)

    # Cost is value indicating possible error in the clusters. Ideal value is 0.0
    if FLAG_VERBOSE:
        cost = km.cost_
        print("*** COST: ***")
        print(cost)

    # Prints 5 largest cluster labels and number of records per cluster.
    if FLAG_VERBOSE:
        most_common = Counter(labels).most_common(5)
        print("Most populated centroids")
        print(most_common)

    # Generate cluster dictionary to be inserted in the centroid_db.
    # Qual_id: qual_id of given documents
    # Features: Dictionary of all possible features in passed documents.
    # Centroids: List of generated centroids.
    # Behavioral_traits: Behavioral traits associated with at least one
    # document assigned to the given centroid.
    # Centroids and behavioral_traits have the same lengths. The behavioral
    # traits in a given index of behavioral_traits is associated with the same
    # index of centroids.
    if FLAG_USE_CENTROID_DB:
        post = {'qual_id': qual_id,
                'features': features,
                'centroids': centroids.tolist(),
                'behavioral_traits': {}}

        # Reuse queried documents.
        data = data.rewind()

        # Add associated behavioral traits to cluster dictionary.
        for doc in data:
            if doc['response'] is None:
                continue
            else:
                temp = np.array([])
                temp = addFeatures(features, temp, doc['response'])
                temp = np.reshape(temp, (-1, num_features))
                label = km.predict(temp)[0]
                if str(label) not in post['behavioral_traits']:
                    post['behavioral_traits'][str(label)] = doc['behavioral_traits']
                X_ids.append(doc['_id'])

        # Add generated cluster dictionary to centroid_db.
        # If a record shares the same qual_id as the generated cluster dictionary,
        # then the stored record will be overwritten.
        print("Posting centroids to database centroids.")
        centroid_db.replace_one({'qual_id': qual_id}, post, upsert=True)
        print(qual_id, "complete.")
        print()

    if FLAG_DO_ANALYSIS:
        # perform some automatic EDA on largest clusters and save
        # collect ids of examples
        data = data.rewind()
        X_ids = []
        for doc in data:
            if doc['response'] is None:
                continue
            else:
                X_ids.append(doc['_id'])
        out_dir = ANALYS_OUT_DIR
        if out_dir is None:
            out_dir = "./out/" + str(qual_id)
        analys = cluster_analyzer(collection, out_dir)
        analys.analyze(student_data, labels, centroids, X_ids, qual_id, interpretFeatures(features, []))


# Cross-reference a passed qual_id with associated centroids.
def predict(qual_id):
    # Find first document with passed qual_id and incorrect answer.
    doc = collection.find_one({"qual_id": qual_id, "correct": False})
    # Find centroids for passed qual_id
    cluster = centroid_db.find_one({"qual_id": qual_id})
    # If cluster does not exist, record will not be cross-referenced.
    # Recommend generating centroids by calling the fit method.
    if cluster is None:
        print("No cluster.")
        return
    centroids = cluster['centroids']
    features = cluster['features']
    behavioral_traits = cluster['behavioral_traits']

    student_data = np.array([])
    # If document has no response field, it cannot be compared to the generated
    # clusters.
    if doc['response'] is None:
        print("No response in document.")
        return
    else:
        student_data = addFeatures(features, student_data, doc['response'])

    # Manually cross-references document to generated centroids. K-modes package
    # has a predict method but it is not used since KModes objects are not
    # stored. Records are associated with a centroid if every feature is
    # identical. If a match is found, the qual_id, cluster traits, and document
    # traits are printed.
    i = 0
    for centroid in centroids:
        j = 0
        for feature in centroid:
            if feature != student_data[j]:
                break
            if j == len(student_data)-1:
                print("Qual_id:", qual_id)
                print("Cluster traits:", behavioral_traits[str(i)])
                print("Document traits:", doc['behavioral_traits'])
                print()
            j += 1
        i += 1

# Method for easy setup
def first():
    f = open('behavioral_traits_per_qual_id.csv', 'r')
    reader = csv.reader(f)
    count = 1
    percentage = 0
    total = sum(1 for row in reader)
    f.seek(0)
    data = centroid_db.find()
    # i = 0
    # for doc in data:
    #     i += 1
    for row in reader:
    #     if count <= i:
    #         count += 1
    #         continue
        if count*100/total >= percentage:
            percentage += 1
        sys.stdout.write("\r|%-73s| %d / %d" % ('#' * int(percentage*.73), count, total))
        fit(row[0], int(row[6]))
        count += 1
        if count == 100:
            break
    f.close()


# Method for easy testing
def second():
    f = open('behavioral_traits_per_qual_id.csv', 'r')
    reader = csv.reader(f)
    count = 0
    for row in reader:
        predict(row[0])
        count += 1
        if count == 100:
            break
    f.close()

# Script to aid in using implementations
# Call "fit" with a qual_id and number of documents with incorrect answers
# to be clustered
# Note: a database called "centroid_db" will be generated which stored the
# centroids and associated information

# Call "predict" with a qual_id to cross-reference a document with the
# generated clusters
if __name__ == '__main__':
    desc = "Run clustering on woot math problems (per-problem clustering). This script expects a QualID, and will perform clustering just on that one problem."
    epil = "If you choose to run additional analysis, this script will print summaries of the most common clusters and download some example screenshots."
    argparser = argparse.ArgumentParser(description=desc, epilog=epil)
    argparser.add_argument('qual_id', help="Qual ID of problem to analyze")
    argparser.add_argument('-k', '--num_clusters', help="number of clusters to find (default:225)",
                           required=False, default=255)
    argparser.add_argument('-n', '--num_ex', help="Number of examples of qual_id to use (default:500)",
                           required=False, default=500)
    argparser.add_argument('-a', '--analyze', help="Perform additional analysis on clusters",
                           action='store_true', required=False)
    argparser.add_argument('-s', '--save', help="Save centroids to database",
                           action='store_true', required=False)
    argparser.add_argument('-o', '--output_dir', help="output directory for analysis. Default: './out/<qual_id>/'",
                           required=False)
    argparser.add_argument('-v', '--verbose', help="Print debug info",
                           action='store_true', required=False)
    args = argparser.parse_args()

    # Run this method first to generate a database of centroids.
    # Method first() requires access to the file called
    # "behavioral_traits_per_qual_id.csv" to read in qual_ids.
    # Current implementations performs a query on the passed qual_id which
    # may take a few minutes per qual_id.
    # The program can be cancelled at anytime. It will continue with the last
    # qual_id it was processing when cancelled.
    #first()

    # Run this method second to test some qual_ids against the generated
    # clusters.
    # Method second() requires access to the file called
    # "behavioral_traits_per_qual_id.csv" to read in qual_ids.
    # second()

    # Methods to manually perform fit and predict operations for any qual_id
    #fit("xSDXuo9OEH.bonus.OG_XxtbnEa", 500)
    #predict("xSDXuo9OEH.bonus.OG_XxtbnEa")

    # fill out script parameters
    if args.num_clusters is not None:
        NUM_CLUSTERS = args.num_clusters
    ANALYS_OUT_DIR = args.output_dir
    FLAG_USE_CENTROID_DB = args.save
    FLAG_VERBOSE = args.verbose
    FLAG_DO_ANALYSIS = args.analyze

    # run
    first()
    # fit(args.qual_id, args.num_ex)
