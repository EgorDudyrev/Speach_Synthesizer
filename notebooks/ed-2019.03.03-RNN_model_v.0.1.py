
# coding: utf-8

# In[1]:


import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

import importlib
if '/opt/notebooks/' not in sys.path:
    sys.path.append('/opt/notebooks/')

try: importlib.reload(sl)
except: import synt_lib as sl


# In[2]:


sess = tf.Session()


# In[3]:


DIRS = sl.get_dirs()
M_PARAMS = sl.get_model_params()


# In[4]:


quant = 256*256#16#M_PARAMS['QUANTIZATION_CHANNELS']


# # Get Data

# In[5]:


n_files = 3


# In[144]:


tf.concat(Xs, axis=0)


# In[149]:


X[1:]


# In[175]:


wav_fnames = Path(DIRS['RAW_DATA']).rglob("*.wav")
Xs = []
Ys = []
for idx, fname in enumerate(wav_fnames):
    if idx==n_files: break
    audio = sl.load_audio_not_one_hot(fname.as_posix(), quantization_channels=quant)
    Xs.append(audio[:-1])
    Ys.append(audio[1:])
    
X = tf.concat(Xs,axis=0)
Y = tf.concat(Ys,axis=0)
print(X.shape, Y.shape)
X[1000:1005].eval(session=sess), Y[1000:1005].eval(session=sess)


# In[176]:


nbits = int(np.log2(quant))
n = 2**(nbits//2)


# In[204]:


X_c, X_f = X//n, X%n
Y_c, Y_f = Y//n, Y%n


# In[205]:


X_c, X_f, Y_c, Y_f = tuple(map(lambda x: (x-128)/128, [X_c, X_f, Y_c, Y_f]))


# In[212]:


X_c[:-1]


# In[240]:





# In[248]:


cur_X = tf.concat([tf.reshape(X_c[:-1], (-1,1)), tf.reshape(X_f[:-1], (-1,1)), tf.reshape(X_c[1:], (-1,1))], axis=1)


# In[249]:


cur_X


# In[259]:


import numpy as np


# In[270]:


c = X_c[1000:1010].eval(session=sess).reshape(-1,1)
f = X_f[1000:1010].eval(session=sess).reshape(-1,1)


# In[273]:


x_t = np.concatenate([c[:-1],f[:-1], c[1:]], axis=1)


# In[274]:


x_t


# In[258]:


c,f


# In[ ]:


x_t = 


# # Build model

# In[20]:


num_epochs = 10
total_series_length = 50000
truncated_backprop_length = 10#M_PARAMS['SAMPLE_RATE']
batch_size = 5
num_batches = 5#total_series_length//batch_size//truncated_backprop_length


# In[ ]:


with tf.name_scope('Model_0.1'):
    batchX_placeholder = tf.placeholder(tf.float32, [None, truncated_backprop_length, quant])
    batchY_placeholder = tf.placeholder(tf.int32, [None, truncated_backprop_length, quant])
    init_state = tf.placeholder(tf.float32, [None, quant])
    
    W = tf.Variable(np.random.rand(quant*2, quant), dtype=tf.float32)
    b = tf.Variable(np.zeros((1,quant)), dtype=tf.float32)
    
    inputs_series = tf.unstack(batchX_placeholder, axis=1)
    labels_series = tf.unstack(batchY_placeholder, axis=1)
    
    # Forward pass
    current_state = init_state
    states_series = []
    for current_input in inputs_series:
        current_input = tf.reshape(current_input, [-1, quant])
        input_and_state_concatenated = tf.concat([current_input, current_state], axis=1)  # Increasing number of columns

        next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
        states_series.append(next_state)
        current_state = next_state
    
    losses = [tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels) for logits, labels in zip(states_series,labels_series)]
    total_loss = tf.reduce_mean(losses)

    train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    for epoch_idx in tqdm_notebook(range(num_epochs),desc='epochs'):
        x,y = tf.concat([X for i in range(batch_size)],axis=0), tf.concat([Y for i in range(batch_size)],axis=0)
        
        _current_state = np.zeros((batch_size, quant))


        for batch_idx in tqdm_notebook(range(num_batches),desc='batches',leave=False):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, states_series],
                feed_dict={
                    batchX_placeholder:batchX.eval(session=sess),
                    batchY_placeholder:batchY.eval(session=sess),
                    init_state:_current_state
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
