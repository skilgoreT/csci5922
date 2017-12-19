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

from pymongo import MongoClient
import json
import numpy as np
# KModes can be found at https://github.com/nicodv/kmodes
from kmodes.kmodes import KModes
from collections import Counter
import csv
import sys

client = MongoClient()
database = client.b
collection = database.anon_student_task_responses
centroid_db = database.centroids


# Create dictionary of features for a passed record
def retrieveKeys(response, features):
    for ex in response:
        if type(response[ex]) is dict:
            features[ex] = {}
            features[ex] = retrieveKeys(response[ex], features[ex])
        elif type(response[ex]) is list:
            continue
        else:
            features[ex] = True
    return features


# Adds features from records to global feature dictionary
def mergeFeatures(doc_features, features):
    for ex in doc_features:
        if type(doc_features[ex]) is dict:
            if ex not in features:
                features[ex] = {}
            features[ex] = mergeFeatures(doc_features[ex], features[ex])
        elif type(doc_features[ex]) is list:
            continue
        elif ex not in features:
            features[ex] = True
    return features


# Add features to a passed record
def addFeatures(features, student_data, response):
    for ex in features:
        if type(features[ex]) is dict:
            if ex not in response:
                response[ex] = {}
            student_data = addFeatures(features[ex], student_data, response[ex])
        else:
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


# Generate centroids given a qual_id and an amount of document to be trained
def fit(qual_id, count):
    # More than 500 documents results in slow training
    if count >= 500:
        count = 500
    # Query for passed qual_id with incorrect answers
    # May take a lengthly amount of time. Recommend optimizing query.
    # print("Querying for", qual_id)
    data = collection.find({"qual_id": qual_id, "correct": False})[:count]
    # print("Query complete.")

    # Compile dictionary of all possible features in given list of records
    # print("Compiling dictionary of features.")
    features = {}
    for doc in data:
        doc_features = {}
        if doc['response'] is None:
            continue
        doc_features = retrieveKeys(doc['response'], doc_features)
        features = mergeFeatures(doc_features, features)
    # print("Feature compilation complete.")

    # Count number of features
    length = countFeatures(features)
    if length == 0:
        return

    # Reuse queried documents.
    data = data.rewind()

    # Append missing features to all records and assign common benign value.
    # Current benign value is an empty string.
    # print("Appending features to documents.")
    student_data = np.array([])
    for doc in data:
        if doc['response'] is None:
            continue
        else:
            temp = np.array([])
            temp = addFeatures(features, temp, doc['response'])
            if len(student_data) == 0:
                student_data = np.append(student_data, temp)
                student_data = np.reshape(student_data, (-1, length))
            else:
                student_data = np.append(student_data, [temp], axis=0)
    # print("Finished appending features to documents.")

    # Perform k-modes clustering
    # print("Clustering...")
    clusters = len(student_data)
    # K-modes implementation can't generate more than 255 centroids
    if clusters > 255:
        clusters = 255
    km = KModes(n_clusters=clusters, init='Cao', n_init=4, verbose=False)
    # print("Finished.")
    km.fit(student_data)

    # Print important information from clustering
    # Centroids are common values to each cluster
    centroids = km.cluster_centroids_
    # print("Centroids")
    # print(centroids)

    # Labels is a list indicating which cluster each record belongs to
    labels = km.labels_
    # print("Labels")
    # print(labels)

    # Cost is value indicating possible error in the clusters. Ideal value is
    # 0.0. If value is greater than 0.0, then the max number of clusters were
    # generated and some responses were assigned to an inexact cluster. This.
    # would result in the largest cluster having having documents it shouldn't.
    # Recommend re-clustering with fewer documents or more clusters if possible.
    cost = km.cost_
    # print("Cost")
    # print(cost)

    # Prints 5 largest cluster labels and number of records per cluster.
    most_common = Counter(labels).most_common(5)
    # print("Most populated centroids")
    # print(most_common)

    # Generate cluster dictionary to be inserted in the centroid_db.
    # Qual_id: qual_id of given documents
    # Features: Dictionary of all possible features in passed documents.
    # Centroids: List of generated centroids.
    # Cluster_sizes: Number of documents in each cluster.
    # Behavioral_traits: Behavioral traits associated with at least one
    # document assigned to the given centroid.
    # Screenshot_urls: A screenshot from one document within each cluster.
    # Centroids and behavioral_traits have the same lengths. The behavioral
    # traits in a given index of behavioral_traits is associated with the same
    # index of centroids.
    post = {'qual_id': qual_id,
            'features': features,
            'centroids': {},
            'cluster_sizes': {},
            'behavioral_traits': {},
            'screenshot_urls': {}}

    for i in Counter(labels).most_common(len(centroids)):
        if str(i[0]) not in post['cluster_sizes']:
            post['cluster_sizes'][str(i[0])] = str(i[1])

    for i in range(len(centroids.tolist())):
        if str(i) not in post['centroids']:
            post['centroids'][str(i)] = centroids.tolist()[i]

    # Reuse queried documents.
    data = data.rewind()
    label = 0
    for doc in data:
        if doc['response'] is None:
            continue
        elif str(labels[label]) not in post['screenshot_urls']:
            post['screenshot_urls'][str(labels[label])] = doc['screenshot_url']
            label += 1
        else:
            label += 1

    # Reuse queried documents.
    data = data.rewind()

    # Add associated behavioral traits to cluster dictionary.
    for doc in data:
        if doc['response'] is None:
            continue
        else:
            temp = np.array([])
            temp = addFeatures(features, temp, doc['response'])
            temp = np.reshape(temp, (-1, length))
            label = km.predict(temp)[0]
            if str(label) not in post['behavioral_traits']:
                post['behavioral_traits'][str(label)] = doc['behavioral_traits']

    # Add generated cluster dictionary to centroid_db.
    # If a record shares the same qual_id as the generated cluster dictionary,
    # then the stored record will be overwritten.
    # print("Posting centroids to database centroids.")
    centroid_db.replace_one({'qual_id': qual_id}, post, upsert=True)
    # print(qual_id, "complete.")
    # print()


# Cross-reference a passed qual_id with associated centroids.
def predict(qual_id):
    # Find first document with passed qual_id and incorrect answer.
    doc = collection.find({"qual_id": qual_id, "correct": False})[2]

    # Find centroids for passed qual_id
    cluster = centroid_db.find_one({"qual_id": qual_id})
    # If cluster does not exist, record will not be cross-referenced.
    # Recommend generating centroids by calling the fit method.
    if cluster is None:
        fit(qual_id, 500)
        cluster = centroid_db.find_one({"qual_id": qual_id})
    if cluster is None:
        print("Qual_id:", qual_id)
        print("No cluster.")
        print()
        return
    features = cluster['features']
    centroids = cluster['centroids']
    cluster_sizes = cluster['cluster_sizes']
    behavioral_traits = cluster['behavioral_traits']
    screenshot_urls = cluster['screenshot_urls']

    student_data = np.array([])
    # If document has no response field, it cannot be compared to the generated
    # clusters.
    if doc['response'] is None:
        print("Qual_id:", qual_id)
        print("No response in document.")
        print()
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
        for feature in centroids[centroid]:
            if feature != student_data[j]:
                break
            if j == len(student_data)-1:
                print("Qual_id:", qual_id)
                print("Cluster traits:", behavioral_traits[str(i)])
                print("Document traits:", doc['behavioral_traits'])
                print("Cluster screenshot_url:", screenshot_urls[str(i)])
                print("Document screenshot_url:", doc['screenshot_url'])
                print()
            j += 1
        i += 1


# Method for easy setup
def first():
    f = open('behavioral_traits_per_qual_id.csv', 'r')
    reader = csv.reader(f)
    count = 0
    percentage = 0
    total = sum(1 for row in reader)
    f.seek(0)
    data = centroid_db.find()
    i = 0
    for doc in data:
        i += 1
    for row in reader:
        if count <= i:
            count += 1
            continue
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
    # Run this method first to generate a database of centroids.
    # Method first() requires access to the file called
    # "behavioral_traits_per_qual_id.csv" to read in qual_ids.
    # Current implementations performs a query on the passed qual_id which
    # may take a few minutes per qual_id.
    # The program can be cancelled at anytime. It will continue with the last
    # qual_id it was processing when cancelled.
    first()

    # Run this method second to test some qual_ids against the generated
    # clusters.
    # Method second() requires access to the file called
    # "behavioral_traits_per_qual_id.csv" to read in qual_ids.
    # second()

    # Methods to manually perform fit and predict operations for any qual_id
    # fit("xSDXuo9OEH.bonus.OG_XxtbnEa", 500)
    # predict("FmeMEBRHmB.bonus.24M2DNktUG")
