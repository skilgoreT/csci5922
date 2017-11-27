# %load read_data.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pymongo import MongoClient

DB_HOST = '10.0.1.3'
DB_NAME = 'wootmathdev'
DB_PORT = 27017

client = MongoClient(DB_HOST, DB_PORT)
db = client[DB_NAME]
data = db['anon_student_task_responses'].find({}, {'student.student_id':1}).limit(1000)
unique_ids = set([d['student']['student_id'] for d in data])

student_vectors = []
for id in unique_ids:
  # all the entries associated with a specific student
  data = db['anon_student_task_responses'].find({'student.student_id':id})
  responses = {}
  for d in data:
    responses[d['qual_id']] = { 'correct':d['correct'], 'ts':d['t'] }

  student_vectors.append(pd.DataFrame(responses))

# The series index is the unique student id, series elements are DataFrames with cols indexed by qual_id
student_vectors = pd.Series(student_vectors, index=unique_ids)
student_vectors[0].head()
