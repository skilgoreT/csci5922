#!/usr/bin/env python

# %load read_data.py
from pymongo import MongoClient
from collections import defaultdict
import re
import os
cwd = os.path.dirname(os.path.realpath(__file__))

DB_HOST = '10.0.1.3'
DB_NAME = 'wootmathdev'
DB_PORT = 27017

# primary query limit (use 0 for all records)
N_LIMIT = 10000
# Limit to N_TASKS
N_TASK = 10
# Student vectors are dropped if they have responsed to less than this proportion of N_TASK
MIN_TASK_PERCENT = 0.5

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
data = db['anon_student_task_responses'].find(has_standards, {'student.student_id':1,
                                                              'ccssm_standards':1,
                                                              'problem_set_id':1,
                                                              'problem_set_subspace':1,
                                                              'qual_id':1}).limit(N_LIMIT)

# populate a set of unique problem_set_id.problem_set_subspace
unique_ids = set([d['student']['student_id'] for d in data])
data.rewind() # reset mongo cursor for next pass

# number of tasks by associated ccssm standard
ccssm_count = defaultdict(int)
# number of tasks by task_id (problem_set_id.problem_set_subspace)
task_count = defaultdict(int)
for d in data:
  for std in d['ccssm_standards']:
    ccssm_count[std] += 1
  # all but last token in qual_id is the task_id
  task_id = re.sub('\.\w*$', '', d['qual_id'])
  task_count[task_id] += 1

task_ids = sorted(task_count, key=task_count.get, reverse=True)[:N_TASK]

print(f"Found {len(ccssm_count)} CCSSM standards in {N_LIMIT} records")
# for std in sorted(ccssm_count, key=ccssm_count.get, reverse=True):
#   print("Standard: {:25} \t Count: {}".format(std, ccssm_count[std]))

print(f"Found {len(task_count)} task ids  in {N_LIMIT} records")
#for task_id in task_ids:
#  print("Task ID: {:25} \t Count: {}".format(task_id, task_count[task_id]))

### Build Student Vectors #############################
## Build the student vectors
#######################################################

# Student vectors are built by selecting student data from the N_TASK tasks
# in the task_ids vector. The task_ids vector is sorted by response count
# so these tasks are the set for which we have the most data. The per task
# query is subject to N_LIMIT (for generating small data sets quickly). The
# data is post processed to remove students that have fewer than 50%

student_vectors = defaultdict(list)
for idx, task_id in enumerate(task_ids):
  problem_set_id, problem_set_subspace = task_id.split('.')
  responses = db['anon_student_task_responses'].find({'problem_set_id':problem_set_id,
                                                      'problem_set_subspace':problem_set_subspace,
                                                      **has_standards}).limit(N_LIMIT)
  print(f"processing task {idx}")
  for r in responses:
    student_id = r['student']['student_id']
    student_vectors[student_id].append({
      'task_id': task_id,                # task_id (problem_set_id.problem_set_subspace)
      'qual_id': r['qual_id'],           # qualified id
      'correct':r['correct'],            # answer correct
      'ts':r['t'],                       # timestamp
      'ccssm': r['ccssm_standards'][0],  # CCSSM
      'untouched': r['untouched'],       # skipped the problem
      'second_try': r['second_try'],     # retry
      'time_spent': r['time_spent']      # dwell time
    })

# build a dictionary from student_id to unique task id count
uniq_task_count = {}
for id in student_vectors.keys():
  uniq_task_ids = set([r['task_id'] for r in student_vectors[id]])
  uniq_task_count[id] = len(uniq_task_ids)

# require that student has seen at least 50% of the tasks in out task set
print(f"N student: {len(student_vectors)}")
# You have to iterate this way to do an in loop manipulation of the dictionary (Python3)
for student_id in list(student_vectors.keys()):
  if uniq_task_count[student_id]< MIN_TASK_PERCENT*N_TASK: del(student_vectors[student_id])
print(f"N student: {len(student_vectors)} after N_TASK cut")

### Write Student Vectors ############################
## JSON & pickle format
######################################################

# as JSON
filepath = f"{cwd}/student_vectors_n_task_{N_TASK}_n_limit_{N_LIMIT}.json"
import json
with open(filepath, 'w') as outfile:
  json.dump(student_vectors, outfile, ensure_ascii=False)
print(f"Wrote student vectors to {filepath}")

# as python binary
filepath = f"{cwd}/student_vectors_n_task_{N_TASK}_n_limit_{N_LIMIT}.pickle"
import pickle
with open(filepath, 'wb') as f:
  pickle.dump(student_vectors, f)
print(f"Wrote student vectors to {filepath}")
