#!/usr/bin/env python

# %load read_data.py
from pymongo import MongoClient
from collections import defaultdict

import os
cwd = os.path.dirname(os.path.realpath(__file__))

DB_HOST = '10.0.1.3'
DB_NAME = 'wootmathdev'
DB_PORT = 27017

# primary query limit (use 0 for all records)
N_LIMIT = 1000000

client = MongoClient(DB_HOST, DB_PORT)
db = client[DB_NAME]

# primary query
# -- require associated ccssm_standard
# -- require student_id
# -- optional limit
has_standards = {'$nor': [
    {'ccssm_standards': {'$exists': False}},
    {'ccssm_standards': {'$size': 0}}
]}
data = db['anon_student_task_responses'].find(has_standards, {'student.student_id':1, 'ccssm_standards':1}).limit(N_LIMIT)

# populate a set of unique student ids
unique_ids = set([d['student']['student_id'] for d in data])
data.rewind() # reset mongo cursor for next pass

# number of tasks by associated ccssm standard
ccssm_count = defaultdict(int)
# number of tasks by associated student_id
id_count = defaultdict(int)
for d in data:
  id = d['student']['student_id']
  id_count[id] += 1
  for std in d['ccssm_standards']:
    ccssm_count[std] += 1

print(f"Found {len(ccssm_count)} CCSSM standards in {N_LIMIT} records")
for std in sorted(ccssm_count, key=ccssm_count.get, reverse=True):
  print("Standard: {:25} \t Count: {}".format(std, ccssm_count[std]))

print(f"Found {len(id_count)} student ids in {N_LIMIT} records")
#for id in sorted(id_count, key=id_count.get, reverse=True):
#  print("ID: {:25} \t Count: {}".format(id, id_count[id]))

### Build Student Vectors ############################
## Build the student vectors for specified constraints
######################################################

# minimum number of completed tasks per student
MIN_NUM_TASK = 100
unique_ids = [id for id in unique_ids if id_count[id] >= MIN_NUM_TASK]
print(f"Processing {len(unique_ids)} students after applying MIN_NUM_TASK = {MIN_NUM_TASK} filter")
student_vectors = []
for id in unique_ids:
  # all the entries associated with a specific student
  data = db['anon_student_task_responses'].find({'student.student_id':id, **has_standards})
  responses = {}
  for d in data:
    responses[d['qual_id']] = {
      'correct':d['correct'],            # answer correct
      'ts':d['t'],                       # timestamp
      'ccssm': d['ccssm_standards'][0],  # CCSSM
      'untouched': d['untouched'],       # skipped the problem
      'second_try': d['second_try']      # correct on retry
    }
  # store student vector in list
  student_vectors.append(responses)

### Write Student Vectors ############################
## JSON & pickle format
######################################################

# as JSON
filepath = f"{cwd}/student_vectors.json"
import json
with open(filepath, 'w') as outfile:
  json.dump(student_vectors, outfile, ensure_ascii=False)
print(f"Wrote student vectors to {filepath}")

# as python binary
filepath = f"{cwd}/student_vectors.pickle"
import pickle
with open(filepath, 'wb') as f:
  pickle.dump(student_vectors, f)
print(f"Wrote student vectors to {filepath}")
