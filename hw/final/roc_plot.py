#!/usr/bin/env python

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import os
cwd = os.path.dirname(os.path.realpath(__file__))

import pickle
filepath = f"{cwd}/concat_roc.pickle"
roc_curves = pickle.load( open(filepath, "rb" ) )
for task_id, roc in roc_curves.items():
  plt.plot(roc[0], roc[1])

