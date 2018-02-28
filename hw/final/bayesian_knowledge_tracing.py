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
test_vectors = get_vectors(test)

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

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def make_stupid_auc(students):
  def stupid_auc(X):
    L0 = X
    p_vector = []
    t_vector = []
    for student_id, encoded_responses in students.items():
      L = L0
      for encoded_response in encoded_responses:
        p_vector.append(L0)
        t_vector.append(encoded_response)
    return p_vector, t_vector
  return stupid_auc

stupid_best_fit = {}
for task_id, students in model_vectors.items():
  cost = make_stupid_cost(students)
  x0 = np.array([0.0])
  res = minimize(cost, x0, method='nelder-mead', options={'xtol': 1e-14, 'disp': False})
  stupid_best_fit[task_id] = res['x']
  #print(res)
roc_curves = {}
stupid_table_data = []
stupid_tpr = {}
for task_id, students in test_vectors.items():
  cost = make_stupid_cost(students)
  test_set_error = cost(stupid_best_fit[task_id])
  stupid_auc = make_stupid_auc(students)
  p_vector, t_vector = stupid_auc(stupid_best_fit[task_id])
  hit = 0.0
  for i in range(0, len(t_vector)):
    if t_vector[i]==1: hit+=1
  hit_rate = hit/len(t_vector)
  t_vector = np.array(t_vector)
  p_vector = np.array(p_vector)
  auc = roc_auc_score(t_vector, p_vector)
  roc_curves[task_id] = {'x':[], 'y':[]}
  roc_curves[task_id]['x'], roc_curves[task_id]['y'], _ = roc_curve(t_vector, p_vector)

  stupid_table_data.append([task_id, len(t_vector)] + stupid_best_fit[task_id].tolist() + [test_set_error, hit_rate, auc])
  stupid_tpr[task_id] = hit_rate

print("\n## Stupid Predictor ##")
from tabulate import tabulate
print(tabulate(stupid_table_data, headers = ['task_id', 'N_test', 'L0', 'Test Set Error', 'PPV -- TP/(TP+FP)', 'AUC'], tablefmt="psql"))

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

def make_bkt_auc(students):
  def bkt_auc(X):
    L0, T, S, G = X
    p_vector = []
    t_vector = []
    for student_id, encoded_responses in students.items():
      L = L0
      for idx, encoded_response in enumerate(encoded_responses):
        p = probability_correct(L,T,S,G)
        t = encoded_response
        p_vector.append(p)
        t_vector.append(t)
        L = update_L(encoded_response, L,T,S,G)
    return p_vector, t_vector
  return bkt_auc

######################################
# AUC
######################################
# y_true = np.array([[0],[1],[1]])
# y_scores = np.array([[0.1], [0.8], [0.9]])
# print("sklearn auc: {}".format(roc_auc_score(y_true, y_scores)))

# filepath = f"{cwd}/bkt_best_fit.json"
# bkt_best_fit = json.load(open(filepath))
# for task_id, students in test_vectors.items():
#   bkt_auc = make_bkt_auc(students)
#   p_vectors, t_vectors = bkt_auc(bkt_best_fit[task_id])
#   t_vectors = np.array(t_vectors)
#   p_vectors = np.array(p_vectors)
#   print("task_id: {} sklearn auc: {}".format(task_id, roc_auc_score(t_vectors, p_vectors)))

bkt_best_fit = {}
for task_id, students in model_vectors.items():
  cost = make_bkt_cost(students)
  x0 = np.array([0.8, 0.1, 0.1, 0.1])
  res = minimize(cost, x0, method='nelder-mead', options={'xtol': 1e-3, 'disp': False})
  bkt_best_fit[task_id] = res['x'].tolist()

bkt_table_data = []
hit_rates = []
aucs = []
roc_curves = {}
for task_id, students in test_vectors.items():
  # best fit error
  cost = make_bkt_cost(students)
  test_set_error = cost(bkt_best_fit[task_id])
  # best fit AUC
  bkt_auc = make_bkt_auc(students)
  p_vector, t_vector = bkt_auc(bkt_best_fit[task_id])
  t_vector = np.array(t_vector)
  p_vector = np.array(p_vector)
  TP=0.0
  FN=0.0
  TN=0.0
  FP=0.0
  for i in range(0, len(p_vector)):
    t = stupid_tpr[task_id]
    if(p_vector[i] > t and t_vector[i] == 1):
      TP+=1
    elif(p_vector[i] < t and t_vector[i] == 1):
      FN+=1
    elif(p_vector[i] > t and t_vector[i] == 0):
      FP+=1
    elif(p_vector[i] < t and t_vector[i] == 0):
      TN+=1
  PPV = TP/(TP+FP)
  TPR = TP/(TP+FN)

  aucs.append(auc)
  try:
    auc = roc_auc_score(t_vector, p_vector)
    roc_curves[task_id] = roc_curve(t_vector, p_vector)
  except:
    auc = 0
  bkt_table_data.append([task_id, len(t_vector)] + bkt_best_fit[task_id] + [test_set_error, f"t={round(stupid_tpr[task_id],2)} -- {round(PPV,4)}", auc])


with open('bkt_best_fit.json', 'w') as outfile:
    json.dump(bkt_best_fit, outfile)

import pickle
with open('roc.pickle', 'wb') as outfile:
    pickle.dump(roc_curves, outfile)

from tabulate import tabulate
print("\n## BKT Predictor ##")
print(tabulate(bkt_table_data, headers=['task_id','N_test', 'L0','T','S','G', 'Test Set Error', 'PPV -- TP/(TP+FP)', 'AUC'], tablefmt="psql"))

print("\nthat is all")