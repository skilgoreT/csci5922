
# coding: utf-8

# In[2]:


#import packages
from __future__ import print_function
import json
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import tensorflow as tf

# this code can work with either tf.VERSION = '1.3.0' or tf.VERSION = '0.12.1', functions may change for other versions
print(tf.VERSION)


# In[3]:


# loading JSON file into dictionary
filepath = "student_vectors_n_task_10_n_limit_10000.json"
student_vectors = json.load(open(filepath))

# examining a case with second attempt
for j in student_vectors['V0D6D2D5O7']:
    if(j['qual_id'] == '1zsCldT4p8.set2.JiTh5Wk0bD'):
        print(j)


# In[4]:


# collect all qual_ids, ccssm_labels, task_ids in separate lists
all_qual_ids = []
all_ccssm_labels = []
all_task_ids = []
for i in student_vectors:
    for j in student_vectors[i]:
        all_qual_ids.append(j['qual_id'])
        all_ccssm_labels.append(j['ccssm'])
        all_task_ids.append(j['task_id'])

# make a set of unique values from the above lists
unique_ids = set(all_qual_ids)
unique_labels = set(all_ccssm_labels)
unique_tasks = set(all_task_ids)
print("Number of unique labels in this dataset " + str(len(unique_labels))) #unique labels
print("Number of unique lessons/tasks in this dataset "+str(len(unique_tasks))) #unique lessons
print("Number of unique questions in this dataset "+str(len(unique_ids))) #this is the length of bit vector (number of unique qual_ids)

# generate vectors to give to fit_transform in multilabelbinarizer to further generate unique 1-hot encoding
transform_ids = []
for i in unique_ids:
    transform_ids.append([i])

transform_labels = []
for i in unique_labels:
    transform_labels.append([i])


# In[5]:


# generate dictionary that maps labels and qual_ids to their respective 1-hot encoding
enc = MultiLabelBinarizer()
qual_ids_1hot = (enc.fit_transform(transform_ids)).astype(float)
qual_ids_classes = enc.classes_
qual_ids_dict = dict(zip(unique_ids, qual_ids_1hot))
labels_1hot = enc.fit_transform(transform_labels).astype(float)
labels_classes = enc.classes_
labels_dict = dict(zip(unique_labels,labels_1hot))


# In[6]:


# generate final encoding
final_encoding = []
second_try_flag = False
for i in student_vectors: #loop over all the students
    interactions_vector = []
    for j in student_vectors[i]: #loop over all the interactions of student 'i'
        #assuming there is no qual_id that has 'untouched': True
        qual_id_vector = qual_ids_dict[j['qual_id']]
        label_vector = labels_dict[j['ccssm']]
        combined_vector = np.concatenate([qual_id_vector,label_vector])
        #scaling the inputs
        if(j['correct'] == True and j['second_try'] == False):
            #the student did correctly in the first try itself!
            combined_vector *= (1.0)
            interactions_vector.append(combined_vector)
        elif(j['correct'] == False and j['second_try'] == False):
            #student will be given second try, nothing will be appended to interactions vector
            second_try_flag = True
        elif(j['correct'] == True and j['second_try'] == True and second_try_flag == True):
            #student does correctly in the second try
            second_try_flag = False
            combined_vector *= (2.0/3.0)
            interactions_vector.append(combined_vector)
        elif(j['correct'] == False and j['second_try'] == True and second_try_flag == True):
            #student did incorrectly the second try
            second_try_flag = False
            combined_vector *= (1.0/3.0)
            interactions_vector.append(combined_vector)
    final_encoding.append(interactions_vector)
# print(final_encoding[0])


# In[7]:


# take input and output vectors from final_encoding (by the rule of o[t] = x[t+1])
final_input_x = []
final_output_y = []
final_seqlen = []
max_interactions = 0 #for max_time to give to RNN
for i in final_encoding: #going per student
    temp_x = []
    temp_max = 0
    for j in i: #going per interaction
        temp_x.append(j)
        temp_max +=1
    final_seqlen.append(temp_max)
    if(max_interactions < temp_max):
        max_interactions = temp_max
    final_output_y.append(j)
    temp_x.pop() #removing last interaction that was appended to x
    final_input_x.append(temp_x)
print("Total Number of Students: " + str(len(final_seqlen))) #number of students
print("Maximum number of interactions (max sequence length): " + str(max_interactions))
# print(len(final_output_y))
# print(type(final_output_y))
# print(final_output_y[0])


# In[41]:


# converting final input to padded input (and ndarray), since we cannot feed lists to RNN since they have inconsistent dimensions.
padded_input_x = np.zeros([len(final_seqlen), max_interactions, len(unique_ids) + len(unique_labels)])
padded_output_y = np.zeros([len(final_seqlen), len(unique_ids) + len(unique_labels)])
for i in range(len(final_input_x)):
    for j in range(len(final_input_x[i])):
        padded_input_x[i][j] = final_input_x[i][j]

padded_output_y = final_output_y

# dividing the dataset into two parts -> training (90%) and testing (10%)
# we have data of about 1255 students as printed above. => 90% ~ 1130 students and 10% ~ 125 students
temp_split = int(np.ceil(0.9 * len(padded_input_x)))
print(temp_split)
train_x = padded_input_x[:temp_split]
train_y = padded_output_y[:temp_split]
train_seqlen = final_seqlen[:temp_split]
test_x = padded_input_x[temp_split:]
test_y = padded_output_y [temp_split:]
test_seqlen = final_seqlen[temp_split:]
#print(type(train_x))
#print(train_x.shape)

#process test_y for auc
for i in range(len(test_y)):
    for j in range(len(test_y[0])):
        if(test_y[i][j] != 0.0 and test_y[i][j] != 1.0):
            test_y[i][j] = round(test_y[i][j])

#prepare for two different AUCs
test_y_ccssm = np.zeros([len(test_y), len(unique_labels)])
test_y_qual = np.zeros([len(test_y), len(unique_ids)])
for i in range(len(test_y)):
    for j in range(len(test_y[0])):
        if j < len(unique_ids):
            test_y_qual[i][j] = test_y[i][j]
        else:
            test_y_ccssm[i][j-len(unique_ids)] = test_y[i][j]
            
test_data = test_x
test_label_ccssm = test_y_ccssm
test_label_qual = test_y_qual


# In[26]:


# Model Parameters
learning_rate = 1.0
display_step = 100
n_hidden = len(unique_ids) + len(unique_labels) #number of hidden units in an RNN cell, we can change this later!
training_steps = 500 #number of epochs
tf.reset_default_graph()

# tf Graph input
x = tf.placeholder(tf.float32, [None, max_interactions, len(unique_ids) + len(unique_labels)]) #(<batch_size>, <max_time>, <num_features>)
y = tf.placeholder(tf.float32, [None, len(unique_ids) + len(unique_labels)]) #(<batch_size>, <num_features>)

def dynamicRNN(x):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    #transformation on outputs needed, otherwise auc=0
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    out_size = len(unique_ids) + len(unique_labels)
    logit = tf.contrib.layers.fully_connected(outputs, out_size, activation_fn=None)
    if tf.VERSION == '1.3.0':
        outputs = tf.nn.sigmoid(logit)
    elif tf.VERSION == '0.12.1': #summit's tensorflow version API doc: https://www.tensorflow.org/versions/r0.12/api_docs/
        outputs = tf.sigmoid(logit)
    return outputs

pred = dynamicRNN(x)
if tf.VERSION == '1.3.0':
    pred_qual,pred_ccssm = tf.split(value=pred,num_or_size_splits=[612,4],axis=1)
elif tf.VERSION == '0.12.1': #summit's tensorflow version API doc: https://www.tensorflow.org/versions/r0.12/api_docs/
    pred_qual,pred_ccssm = tf.split_v(value=pred,size_splits=[612,4],split_dim=1)

# Define loss and optimizer
if tf.VERSION == '1.3.0':
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
elif tf.VERSION == '0.12.1': #summit's tensorflow version API doc: https://www.tensorflow.org/versions/r0.12/api_docs/
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, targets=y))
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(cost)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model - use AUC to evaluate model
if tf.VERSION == '1.3.0':
    auc,  opts = tf.metrics.auc(labels = test_label_qual, predictions = pred_qual, curve='ROC')
    auc_ccssm,  opts_ccssm = tf.metrics.auc(labels = test_label_ccssm, predictions = pred_ccssm, curve='ROC')
elif tf.VERSION == '0.12.1': #summit's tensorflow version API doc: https://www.tensorflow.org/versions/r0.12/api_docs/
    auc,  opts = tf.contrib.metrics.streaming_auc(labels = test_label_qual, predictions = pred_qual, curve='ROC')
    auc_ccssm,  opts_ccssm = tf.contrib.metrics.streaming_auc(labels = test_label_ccssm, predictions = pred_ccssm, curve='ROC')

# Start training
with tf.Session() as sess:
    
    # Initialize the variables (i.e. assign their default value)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for step in range(1, training_steps+1):
        batch_x = train_x
        batch_y = train_y
        
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        test_data = test_x
        test_label = test_y
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            loss= sess.run([cost], feed_dict={x: batch_x, y: batch_y})
            print("Step " + str(step) + ", Loss= " + str(loss))
    print("Optimization Finished!")

    # Calculate test auc
    temp_auc_ccssm, temp_opts_ccssm = sess.run([auc_ccssm,  opts_ccssm], feed_dict={x: test_data, y: test_y})
    temp_auc_qual, temp_opts_qual = sess.run([auc,opts], feed_dict={x: test_data, y: test_y})
    print("Testing auc for qual: " + str(temp_auc_qual) + ", " + str(temp_opts_qual))
    print("Testing auc for ccssm: " + str(temp_auc_ccssm) + ", " + str(temp_opts_ccssm))


# In[ ]:


# Project guidelines: https://www.cs.colorado.edu/~mozer/Teaching/syllabi/DeepLearningFall2017/assignments/assignment7.html
# Resources:
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py


# In[ ]:


# file  = open('debug.txt', 'w')
# file.write(str(debug_list))

