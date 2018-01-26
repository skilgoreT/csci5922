#!/usr/bin/env python

import os
cwd = os.path.dirname(os.path.realpath(__file__))
import json
import math
TEST_PERCENT = 0.2

import numpy as np

# from itertools recipes -- used to split the input data into model and verify sets
from itertools import islice
def take(n, iterable, from_end=False):
    "Return first n items of the iterable as a list"
    if from_end:
      return list(islice(iterable, len(iterable)-n, None))
    else:
      return list(islice(iterable, n))

# correct on second try counts as wrong for now
def encode_response(response):
  if response['correct'] and not response['second_try']:
    return 1
  elif not response['correct'] and not response['second_try']:
    return 0;
  else:
    return None;

filepath = f"{cwd}/student_vectors_n_task_10_n_limit_10000.json"
student_vectors = json.load(open(filepath))
n_student = len(student_vectors)
print(f"read JSON from {filepath}, N: {len(student_vectors)}")
n_test = math.floor(n_student * TEST_PERCENT)
n_model = n_student - n_test
print(f"Splitting {n_student} Students into {n_model} model and {n_test} test")
model = take(n_model, student_vectors.items())
test = take(n_test, student_vectors.items(), from_end=True)


def get_vectors(data):
  vectors = {}
  for student_id, responses in data:
    for response in responses:
      task_id = response['task_id']
      vectors.setdefault(task_id, {})
      vectors[task_id].setdefault(student_id, [])
      encoded_response = encode_response(response)
      if encoded_response == None: continue
      vectors[task_id][student_id].append(encoded_response)
  return vectors

model_vectors = get_vectors(model)
test_vectors = get_vectors(model)

######################################
# Stupid Predictor
######################################

import numpy as np
from scipy.optimize import minimize

def make_stupid_cost(students):
  def stupid_cost(X):
    L0 = X
    E = 0.0
    N = 0.0
    for student_id, encoded_responses in students.items():
      for encoded_response in encoded_responses:
        N += 1.0
        E += (L0 - encoded_response)**2
    E /= N
    return E

  return stupid_cost

stupid_best_fit = {}
for task_id, students in model_vectors.items():
  cost = make_stupid_cost(students)
  x0 = np.array([0.0])
  res = minimize(cost, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
  stupid_best_fit[task_id] = res['x']
  #print(res)

stupid_table_data = []
for task_id, students in test_vectors.items():
  cost = make_stupid_cost(students)
  test_set_error = cost(stupid_best_fit[task_id])
  stupid_table_data.append([task_id] + stupid_best_fit[task_id].tolist() + [test_set_error])

print("\n## Stupid Predictor ##")
from tabulate import tabulate
print(tabulate(stupid_table_data, headers = ['task_id', 'L0', 'Test Set Error']))

######################################
# Bayesian Knowledge Tracing Predictor
######################################

def update_L(correct,L,T,S,G):
  if correct:
    L = L*(1-S)/(L*(1-S) + (1-L)*G)
  else:
    L = L*S/(L*S + (1-L)*(1-G))
  return L + (1-L)*T

def probability_correct(L,T,S,G):
  return L*(1-S) + (1-L)*G

def make_bkt_cost(students):
  def bkt_cost(X):
    L0, T, S, G = X
    E = 0.0
    N = 0.0
    for student_id, encoded_responses in students.items():
      L = L0
      for encoded_response in encoded_responses:
        N += 1.0
        E += (probability_correct(L,T,S,G) - encoded_response)**2
        #print(f"L before {L}")
        L = update_L(encoded_response, L,T,S,G)
        #print(f"L after {L}")
    E /= N
    #print(E)
    return E
  return bkt_cost

bkt_best_fit = {}
for task_id, students in model_vectors.items():
  cost = make_bkt_cost(students)
  x0 = np.array([0.8, 0.1, 0.1, 0.1])
  res = minimize(cost, x0, method='nelder-mead', options={'xtol': 1e-3, 'disp': False})
  bkt_best_fit[task_id] = res['x']

bkt_table_data = []
for task_id, students in test_vectors.items():
  cost = make_bkt_cost(students)
  test_set_error = cost(bkt_best_fit[task_id])
  bkt_table_data.append([task_id] + bkt_best_fit[task_id].tolist() + [test_set_error])

  # print(res)
from tabulate import tabulate

print("\n## BKT Predictor ##")
print(tabulate(bkt_table_data, headers=['task_id','L0','T','S','G']))

print("\nthat is all")