
# coding: utf-8

# In[1]:


import sys
import os

import numpy as np
from numpy import random
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from IPython import display
get_ipython().run_line_magic('matplotlib', 'inline')
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
nbits = int(np.log2(quant))


# # Get Data

# Всего файлов:
# * Обучение (cv-valid-train): 391552
# * cv-valid-dev: 8152
# * cv-valid-test: 7990

# In[5]:


n_files = 3


# In[6]:


wav_fnames = Path(DIRS['RAW_DATA']).rglob("*.wav")
Xs = []
for idx, fname in enumerate(wav_fnames):
    if idx==n_files: break
    audio = sl.load_audio_not_one_hot(fname.as_posix(), quantization_channels=quant)
    Xs.append(audio[:-1])
    
X = tf.concat(Xs,axis=0)
X = tf.reshape(X, (1,-1,1))
n = 2**(nbits//2)
X = tf.concat([X//n, X%n], 2)
X = (X-128)/128
print(X.shape)
X[0,1000:1005].eval(session=sess)


# In[7]:


def generate_batch(X, batch_size, truncated_len):
    idxs = random.randint(0,X.shape[1]-truncated_len-1, size=batch_size)
    x = tf.concat([X[:,idxs[i]:idxs[i]+truncated_len] for i in range(batch_size)],axis=0)
    y = tf.concat([X[:,idxs[i]+1:idxs[i]+truncated_len+1] for i in range(batch_size)],axis=0)
    return x,y


# # Build model

# In[8]:


class GRU:
    """Implementation of a Gated Recurrent Unit (GRU) as described in [1].
    
    [1] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
    
    Arguments
    ---------
    input_dimensions: int
        The size of the input vectors (x_t).
    hidden_size: int
        The size of the hidden layer vectors (h_t).
    dtype: obj
        The datatype used for the variables and constants (optional).
    """
    
    def __init__(self, input_dimensions, hidden_size, dtype=tf.float64):
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Wr = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wr')
        self.Wz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wz')
        self.Wh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wh')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Ur = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ur')
        self.Uz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uz')
        self.Uh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uh')
        
        # Biases for hidden vectors of shape (hidden_size,)
        self.br = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='br')
        self.bz = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='bz')
        self.bh = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='bh')
        
        # Define the input layer placeholder
        self.input_layer = tf.placeholder(dtype=tf.float64, shape=(None, None, input_dimensions), name='input')
        
        # Put the time-dimension upfront for the scan operator
        self.x_t = tf.transpose(self.input_layer, [1, 0, 2], name='x_t')
        
        # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
        self.h_0 = tf.matmul(self.x_t[0, :, :], tf.zeros(dtype=tf.float64, shape=(input_dimensions, hidden_size)), name='h_0')
        
        # Perform the scan operator
        self.h_t_transposed = tf.scan(self.forward_pass, self.x_t, initializer=self.h_0, name='h_t_transposed')
        
        # Transpose the result back
        self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')

    def forward_pass(self, h_tm1, x_t):
        """Perform a forward pass.
        
        Arguments
        ---------
        h_tm1: np.matrix
            The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix
            The input vector.
        """
        # Definitions of z_t and r_t
        z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
        r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)
        
        # Definition of h~_t
        h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)
        
        # Compute the next hidden state
        h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
        
        return h_t


# In[9]:


# The input has 2 dimensions: dimension 0 is reserved for the first term and dimension 1 is reverved for the second term
input_dimensions = 2

# Arbitrary number for the size of the hidden state
hidden_size = 2


# ## V1

# In[13]:


batch_size = 10
truncated_len = M_PARAMS['SAMPLE_RATE']
total_series_length = int(X.shape[1])
num_epochs = 10#400#total_series_length//batch_size//truncated_len
print(batch_size, truncated_len, total_series_length, num_epochs)


# 10 secs per iteration => 1 min per 6 iters => 1 hour per 360 iters

# In[14]:


# Create a new instance of the GRU model
gru = GRU(input_dimensions, hidden_size)


# In[15]:


output = gru.h_t
# Create a placeholder for the expected output
expected_output = tf.placeholder(dtype=tf.float64, shape=(batch_size, truncated_len, 2), name='expected_output')
# Just use quadratic loss
loss = tf.reduce_sum(0.5 * tf.pow(output - expected_output, 2)) / float(batch_size)
# Use the Adam optimizer for training
train_step = tf.train.AdamOptimizer().minimize(loss)


# In[16]:


# Initialize all the variables
init_variables = tf.global_variables_initializer()
sess.run(init_variables)


# In[17]:


# Initialize the losses
train_losses = []
validation_losses = []

# Perform all the iterations
for epoch in tqdm_notebook(range(num_epochs)):
    X_train, Y_train = generate_batch(X, batch_size, truncated_len)
    X_test, Y_test = generate_batch(X, batch_size, truncated_len)
    X_train, Y_train, X_test, Y_test = sess.run([X_train, Y_train, X_test, Y_test])
    
    # Compute the losses
    _, train_loss = sess.run([train_step, loss], feed_dict={gru.input_layer: X_train, expected_output: Y_train})
    validation_loss = sess.run(loss, feed_dict={gru.input_layer: X_test, expected_output: Y_test})
    
    # Log the losses
    train_losses += [train_loss]
    validation_losses += [validation_loss]
    
    # Display an update every 50 iterations
    if epoch % 50 == 0:
        plt.plot(train_losses, '-b', label='Train loss')
        plt.plot(validation_losses, '-r', label='Validation loss')
        plt.legend(loc=0)
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
        print('Iteration: %d, train loss: %.4f, test loss: %.4f' % (epoch, train_loss, validation_loss))


# # Sound generation

# In[21]:


M_PARAMS['SAMPLE_RATE']


# In[22]:


for i in range(len(Xs)):
    t = Xs[i]
    scale = M_PARAMS['SAMPLE_RATE']//50 # 1seconds divide by 50 parts
    plt.plot(sess.run([tf.reduce_mean(t[idx*scale:(idx+1)*scale]) 
                       for idx in range(int(t.shape[0])//scale)]), label=i)
plt.ylim(0, 256*256)
plt.show()


# In[23]:


num_pieces = 1
generated = np.array([0,-1]*num_pieces).reshape(num_pieces,1,2)
generated


# In[24]:


for i in tqdm_notebook(range(M_PARAMS['SAMPLE_RATE']*4)): # 1 seconds of 'speach'
    curX = generated[:,-1,:].reshape(num_pieces,-1,2)
    curY = sess.run(output, feed_dict={gru.input_layer: curX})
    generated = np.concatenate([generated, curY],axis=1)


# In[25]:


gen_to_wav = generated*128+128
gen_to_wav = np.int32((gen_to_wav[:,:,0]*256+gen_to_wav[:,:,1]).round())
gen_to_wav = tf.convert_to_tensor(gen_to_wav)


# In[26]:


plt.plot(audio.eval(session=sess), label='real')
plt.plot(gen_to_wav[0].eval(session=sess), label='generated')
plt.plot(np.int32([np.sin(x/1000)*16000+32256 for x in range(audio.shape[0])]))


# In[27]:


sl.write_audio_not_one_hot(audio=gen_to_wav[0], filename='output_0.wav', session=sess, quantization_channels=quant)


# ## V2

# In[23]:


batch_size = 50
truncated_len = M_PARAMS['SAMPLE_RATE']
total_series_length = int(X.shape[1])
num_epochs = 200#total_series_length//batch_size//truncated_len
print(batch_size, truncated_len, total_series_length, num_epochs)


# 10 secs per iteration => 1 min per 6 iters => 1 hour per 360 iters

# In[24]:


# Create a new instance of the GRU model
gru = GRU(input_dimensions, hidden_size)


# In[25]:


output = gru.h_t
# Create a placeholder for the expected output
expected_output = tf.placeholder(dtype=tf.float64, shape=(batch_size, truncated_len, 2), name='expected_output')
# Just use quadratic loss
loss = tf.reduce_sum(0.5 * tf.pow(output - expected_output, 2)) / float(batch_size)
# Use the Adam optimizer for training
train_step = tf.train.AdamOptimizer().minimize(loss)


# In[26]:


# Initialize all the variables
init_variables = tf.global_variables_initializer()
sess.run(init_variables)


# In[ ]:


# Initialize the losses
train_losses = []
validation_losses = []

# Perform all the iterations
for epoch in tqdm_notebook(range(num_epochs)):
    X_train, Y_train = generate_batch(X, batch_size, truncated_len)
    X_test, Y_test = generate_batch(X, batch_size, truncated_len)
    X_train, Y_train, X_test, Y_test = sess.run([X_train, Y_train, X_test, Y_test])
    
    # Compute the losses
    _, train_loss = sess.run([train_step, loss], feed_dict={gru.input_layer: X_train, expected_output: Y_train})
    validation_loss = sess.run(loss, feed_dict={gru.input_layer: X_test, expected_output: Y_test})
    
    # Log the losses
    train_losses += [train_loss]
    validation_losses += [validation_loss]
    
    # Display an update every 50 iterations
    if epoch % 50 == 0:
        plt.plot(train_losses, '-b', label='Train loss')
        plt.plot(validation_losses, '-r', label='Validation loss')
        plt.legend(loc=0)
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
        print('Iteration: %d, train loss: %.4f, test loss: %.4f' % (epoch, train_loss, validation_loss))


# # Sound generation

# In[ ]:


M_PARAMS['SAMPLE_RATE']


# In[ ]:


for i in range(len(Xs)):
    t = Xs[i]
    scale = M_PARAMS['SAMPLE_RATE']//50 # 1seconds divide by 50 parts
    plt.plot(sess.run([tf.reduce_mean(t[idx*scale:(idx+1)*scale]) 
                       for idx in range(int(t.shape[0])//scale)]), label=i)
plt.ylim(0, 256*256)
plt.show()


# In[ ]:


num_pieces = 1
generated = np.array([0,-1]*num_pieces).reshape(num_pieces,1,2)
generated


# In[ ]:


for i in tqdm_notebook(range(M_PARAMS['SAMPLE_RATE']*4)): # 1 seconds of 'speach'
    curX = generated[:,-1,:].reshape(num_pieces,-1,2)
    curY = sess.run(output, feed_dict={gru.input_layer: curX})
    generated = np.concatenate([generated, curY],axis=1)


# In[ ]:


gen_to_wav = generated*128+128
gen_to_wav = np.int32((gen_to_wav[:,:,0]*256+gen_to_wav[:,:,1]).round())
gen_to_wav = tf.convert_to_tensor(gen_to_wav)


# In[ ]:


plt.plot(audio.eval(session=sess), label='real')
plt.plot(gen_to_wav[0].eval(session=sess), label='generated')
plt.plot(np.int32([np.sin(x/1000)*16000+32256 for x in range(audio.shape[0])]))


# In[ ]:


sl.write_audio_not_one_hot(audio=gen_to_wav[0], filename='output_1.wav', session=sess, quantization_channels=quant)

