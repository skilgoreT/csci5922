#!/usr/bin/env python

import os
cwd = os.path.dirname(os.path.realpath(__file__))

import json
filepath = f"{cwd}/student_vectors_n_task_10_n_limit_10000.json"
student_vectors = json.load(open(filepath))
print(f"read JSON from {filepath}, N: {len(student_vectors)}")

import pickle
filepath = f"{cwd}/student_vectors_n_task_10_n_limit_10000.json"
student_vectors = pickle.load( open(filepath, "rb" ) )
print(f"read pickle from {filepath}, N: {len(student_vectors)}")