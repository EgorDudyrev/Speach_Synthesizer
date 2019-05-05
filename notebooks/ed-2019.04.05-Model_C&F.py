
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
import seaborn as sns

from IPython import display
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook

PARENT_DIR = os.path.realpath('..')
import importlib
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

try: importlib.reload(sl)
except: import synt_lib as sl


# In[2]:


DIRS = sl.get_dirs(parent_dir=PARENT_DIR)
M_PARAMS = sl.get_model_params()


# In[3]:


quant = M_PARAMS['QUANTIZATION_CHANNELS']
nbits = int(np.log2(quant))


# # Build model

# In[4]:


class WaveGRU:
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
    
    def __init__(self, input_dimensions, hidden_size, dtype=tf.float64, variables_values_dict=None):
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        self.define_constants()
        if variables_values_dict is None:
            self.define_variables(dtype)
        else:
            self.restore_variables(variables_values_dict)
        self.define_arithmetics()
        self.define_train_variables()
    
    def define_constants(self):
        # Mask for masking W matrixes
        M = np.ones(shape=(self.input_dimensions, self.hidden_size))
        M[2,:self.hidden_size//2]=0
        self.M = tf.constant(shape=(self.input_dimensions, self.hidden_size), value=M)
        
    def define_variables(self, dtype):     
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Wr = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wr')
        self.Wu = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wu')
        self.We = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='We')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Ur = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ur')
        self.Uu = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uu')
        self.Ue = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ue')
        
        # Biases for hidden vectors of shape (hidden_size,)
        self.br = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='br')
        self.bu = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='bu')
        self.be = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='be')
        
        # O's matrices
        self.O1 = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size//2,self.hidden_size//2), mean=0, stddev=0.01), name='O1')
        self.O3 = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size//2,self.hidden_size//2), mean=0, stddev=0.01), name='O3')
        self.O2 = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size//2,self.hidden_size//2), mean=0, stddev=0.01), name='O2')
        self.O4 = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size//2,self.hidden_size//2), mean=0, stddev=0.01), name='O4')
    
    def restore_variables(self, variables):
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Wr = tf.Variable(variables['Wr:0'], name='Wr')
        self.Wu = tf.Variable(variables['Wu:0'], name='Wu')
        self.We = tf.Variable(variables['We:0'], name='We')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Ur = tf.Variable(variables['Ur:0'], name='Ur')
        self.Uu = tf.Variable(variables['Uu:0'], name='Uu')
        self.Ue = tf.Variable(variables['Ue:0'], name='Ue')
        
        # Biases for hidden vectors of shape (hidden_size,)
        self.br = tf.Variable(variables['br:0'], name='br')
        self.bu = tf.Variable(variables['bu:0'], name='bu')
        self.be = tf.Variable(variables['be:0'], name='be')
        
        # O's matrices
        self.O1 = tf.Variable(variables['O1:0'], name='O1')
        self.O3 = tf.Variable(variables['O2:0'], name='O3')
        self.O2 = tf.Variable(variables['O3:0'], name='O2')
        self.O4 = tf.Variable(variables['O4:0'], name='O4')
    
    def define_arithmetics(self):
        # Define the input layer placeholder
        self.input_layer = tf.placeholder(dtype=tf.float64, shape=(None, None, self.input_dimensions), name='input')
        #[c_t-1, f_t-1, c_t]
        
        # Put the time-dimension upfront for the scan operator
        self.x_t = tf.transpose(self.input_layer, [1, 0, 2], name='x_t')
        #[f_t-1, c_t-1, c_t]
        
        # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
        self.h_0 = tf.matmul(self.x_t[0, :, :], tf.zeros(dtype=tf.float64, shape=(self.input_dimensions, self.hidden_size)), name='h_0')
        
        # Perform the scan operator
        self.h_t_transposed = tf.scan(self.forward_pass, self.x_t, initializer=self.h_0, name='h_t_transposed')
        
        self.y_c, self.y_f = tf.split(self.h_t_transposed, num_or_size_splits=2, axis=2)
        # Transpose the result back
        #self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')
        
        self.P_ct = tf.scan(self.get_P_cs, self.y_c, name='calc_Pc')
        self.c_t_transposed = tf.reduce_max(self.P_ct, axis=2)
        self.c_t = tf.transpose(self.c_t_transposed)
        self.P_ft = tf.scan(self.get_P_fs, self.y_f, name='calc_Pf')
        self.f_t_transposed = tf.reduce_max(self.P_ft, axis=2)
        self.f_t = tf.transpose(self.f_t_transposed)
        
        self.y = tf.stack([self.c_t, self.f_t], axis=2)
    
    def define_train_variables(self):
        self.output = self.y
        self.expected_output = tf.placeholder(
            dtype=tf.float64, shape=(None, None, 2), name='expected_output'
            #(batch_size, truncated_len, 2), name='expected_output'
        )
        #self.loss = tf.reduce_sum(0.5 * tf.pow(self.output - self.expected_output, 2)) / float(batch_size)
        # mean(1/2 * (y-y_true)^2)
        self.loss = tf.reduce_mean(0.5 * tf.pow(self.output - self.expected_output, 2))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        
    def get_P_cs(self, lastP, y_c):
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_c, self.O1)), self.O2), axis=1)
    def get_P_fs(self, lastP, y_f):
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_f, self.O3)), self.O4), axis=1)
        
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
        u_t = tf.sigmoid(tf.matmul(h_tm1, self.Uu) + tf.matmul(x_t, tf.multiply(self.Wu, self.M)) + self.bu)
        r_t = tf.sigmoid(tf.matmul(h_tm1, self.Ur) + tf.matmul(x_t, tf.multiply(self.Wr, self.M)) + self.br)
        # Definition of h~_t
        e_t = tf.tanh(tf.multiply(r_t, tf.matmul(h_tm1, self.Ue))                      +tf.matmul(x_t, tf.multiply(self.We, self.M))                      + self.be)
        # Compute the next hidden state
        h_t = tf.multiply(u_t, h_tm1) + tf.multiply(1 - u_t, e_t)
        return h_t
    
    def train(self, X_train, Y_train, X_test, Y_test, session):
        c_t = session.run(self.c_t, feed_dict={self.input_layer: X_train})
        X_train[:,:,2] = c_t
        # Compute the losses
        _, train_loss = session.run([self.train_step, self.loss],
                                 feed_dict={self.input_layer: X_train, self.expected_output: Y_train})
        validation_loss = session.run(self.loss,
                                   feed_dict={self.input_layer: X_test, self.expected_output: Y_test})
        return train_loss, validation_loss
    
    def validate(self, X_val, Y_val, session):
        c_t = session.run(self.c_t, feed_dict={self.input_layer: X_val})
        X_val[:,:,2] = c_t
        validation_loss = session.sun(self.loss,
                                     feed_dict={self.input_layer: X_val, self.expected_output: Y_val})
        return validation_loss
    
    def generate_sound(self, num_pieces, n_seconds, session, sample_rate=16000):
        generated = np.array([0]*2*num_pieces).reshape(num_pieces,1,2)
        curX = generated[:,-1,:].reshape(num_pieces,-1,2)
        for i in tqdm_notebook(range(sample_rate*n_seconds)): # 1 seconds of 'speach'
            curX = generated[:,-1,:].reshape(num_pieces,-1,2)
            curX = np.concatenate([curX,np.array([[[0]]]*num_pieces) ],axis=2)
            c_t = session.run(self.c_t, feed_dict={self.input_layer: curX})
            curX[:,:,2] = c_t
            curY = session.run(self.output, feed_dict={self.input_layer: curX})
            generated = np.concatenate([generated, curY],axis=1)
        gen_to_wav = generated*128+128
        gen_to_wav = np.int32((gen_to_wav[:,:,0]*256+gen_to_wav[:,:,1]).round())
        gen_to_wav = tf.convert_to_tensor(gen_to_wav)
        return gen_to_wav


# In[54]:


def train_model(input_dimensions, hidden_size, batch_size, truncated_len, num_epochs, model_name,
print_period=50, save_period=50, log_period=50):
    if model_name not in os.listdir(DIRS['MODELS']):
        os.mkdir(DIRS['MODELS']+model_name)
    
    tf.reset_default_graph()
    model = WaveGRU(input_dimensions, hidden_size)
    init_variables = tf.global_variables_initializer()
    saver = tf.train.Saver()
    wav_fnames = Path(DIRS['RAW_DATA']).rglob("*.wav")
    
    epochs_per_files_last = 0
    
    
    # Initialize the losses
    train_losses = []
    validation_losses = []


    with tf.Session() as sess:
        sess.run(init_variables)
        
        # Perform all the iterations
        for epoch in tqdm_notebook(range(num_epochs)):
            if epochs_per_files_last==0:
                X = sl.load_data(wav_fnames, 5)
                total_series_length = int(X.shape[1])
                epochs_per_files_last = total_series_length//batch_size//truncated_len
            epochs_per_files_last-=1
            
            X_train, Y_train, X_test, Y_test = sl.get_train_test(X, batch_size, truncated_len, sess)
            train_loss, validation_loss = model.train(X_train, Y_train, X_test, Y_test, sess)

            # Log the losses
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            # Display an update every 50 iterations
            if epoch % print_period == 0 and epoch!=0:
                print(f'Iteration: {epoch}, train loss: {train_loss:.4f}, val loss: {validation_loss:.4f}')
            if epoch % print_period == 0 and epoch!=0:
                sl.plot_losses(train_losses, validation_losses,
                            title=f'Iteration: {epoch}, train loss: {train_loss:.4f}, val loss: {validation_loss:.4f}')
                plt.show()
            if epoch % save_period == 0:
                saver.save(sess, DIRS['MODELS']+model_name+'/checkpoint',global_step=epoch,write_meta_graph=True)
        
        sl.plot_losses(train_losses, validation_losses,
                     title='Iteration: %d, train loss: %.4f, test loss: %.4f' % (epoch, train_loss, validation_loss))
        plt.show()

        saver.save(sess, DIRS['MODELS']+model_name+'/final')
        
    return train_losses, validation_losses, model


# # Get Data

# Всего файлов:
# * Обучение (cv-valid-train): 391552
# * cv-valid-dev: 8152
# * cv-valid-test: 7990

# In[6]:


wav_fnames = Path(DIRS['RAW_DATA']).rglob("*.wav")


# ## Grid search

# In[7]:


grid_search_res_ds = pd.DataFrame()


# In[42]:


input_dimensions_vars = [3]
hidden_size_vars = [16,128,512,1024]
batch_size_vars = [50,100,200,500,1000]
truncated_len_vars = [10, 100, 200, 500, 1000]


# In[9]:


from itertools import product
from random import shuffle


# In[45]:


all_vars = list(product(input_dimensions_vars, hidden_size_vars, batch_size_vars, truncated_len_vars))
n_vars = len(all_vars)
shuffle(all_vars)

models_dict = {}
# In[ ]:


for idx, p in tqdm_notebook(enumerate(all_vars), total=n_vars):
    input_dimensions, hidden_size, batch_size, truncated_len = p
    model_name = f'C&F_inpdim{input_dimensions}_hsize{hidden_size}_bsize{batch_size}_tlen{truncated_len}'
    print(idx, model_name)
    if model_name in models_dict.keys():
        continue
    train_losses, validation_losses, model = train_model(
        input_dimensions, hidden_size, batch_size, truncated_len, num_epochs=1000,
        model_name=model_name, print_period=1000
    )
    grid_search_res_ds[model_name+'_train'] = train_losses
    grid_search_res_ds[model_name+'_valid'] = validation_losses
    models_dict[model_name] = model

plt.figure(figsize=(10,10))
models =list(models_dict.keys())
for midx, m in enumerate(models):
    plt.subplot(len(models),1,midx+1)
    plt.plot(grid_search_res_ds[m+'_train'], label='Train' if midx==0 else '')
    plt.plot(grid_search_res_ds[m+'_valid'], label='Validation' if midx==0 else '')
    plt.title(m)
plt.figlegend()
plt.tight_layout()
plt.show()
# # Train best model

# In[ ]:


# Initialize the losses
train_losses = []
validation_losses = []


with tf.Session() as sess:
    sess.run(init_variables)
    O1_before = gru.O1.eval(session=sess)
    
    # Perform all the iterations
    for epoch in tqdm_notebook(range(num_epochs)):
        X_train, Y_train, X_test, Y_test = sl.get_train_test(X, batch_size, truncated_len, sess)
        train_loss, validation_loss = gru.train(X_train, Y_train, X_test, Y_test, sess)

        # Log the losses
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        # Display an update every 50 iterations
        if epoch % 100 == 0:
            sl.plot_losses(train_losses, validation_losses,
                        title='Iteration: %d, train loss: %.4f, test loss: %.4f' % (epoch, train_loss, validation_loss))
            plt.show()
            saver.save(sess, DIRS['MODELS']+model_name+'/checkpoint',global_step=epoch,write_meta_graph=False)
    else:
        sl.plot_losses(train_losses, validation_losses,
                    title='Iteration: %d, train loss: %.4f, test loss: %.4f' % (epoch, train_loss, validation_loss))
        plt.show()
        
        saver.save(sess, DIRS['MODELS']+model_name+'/final')
    
    O1_after = gru.O1.eval(session=sess)


# In[15]:


sl.plot_losses(train_losses, validation_losses,
              title='Iteration: %d, train loss: %.4f, test loss: %.4f' % (epoch, train_loss, validation_loss))


# # Restoring model

# In[14]:


tf.reset_default_graph()
saver = tf.train.import_meta_graph(DIRS['MODELS']+model_name+'/final.meta')
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint(DIRS['MODELS']+model_name))
    restored_variables = {x.name:x.eval(session=sess) for x in tf.global_variables()[:13]}


# In[15]:


tf.reset_default_graph()
gru = WaveGRU(input_dimensions, hidden_size, variables_values_dict=restored_variables)


# In[16]:


X = sl.load_data(wav_fnames, 3)


# In[17]:


batch_size = 10
truncated_len = 100#M_PARAMS['SAMPLE_RATE']
total_series_length = int(X.shape[1])
num_epochs = 50#400#total_series_length//batch_size//truncated_len
print(batch_size, truncated_len, total_series_length, num_epochs)


# In[18]:


init_variables = tf.global_variables_initializer()


# In[19]:


with tf.Session() as sess:
    sess.run(init_variables)
    O1_restored = gru.O1.eval(session=sess)


# In[20]:


plt.figure(figsize=(15,4))
for idx, O in enumerate([('before training',O1_before),
                         ('after training', O1_after),
                         ('restored', O1_restored)]):
    title, O = O
    plt.subplot(1,3,idx+1)
    sns.heatmap(O, center=0, cmap='RdBu_r')
    plt.title(title)
plt.tight_layout()
plt.show()


# # Sound generation

# In[21]:


with tf.Session() as sess:
    sess.run(init_variables)
    gen_to_wav = gru.generate_sound(num_pieces=1, n_seconds=2, session=sess, sample_rate=M_PARAMS['SAMPLE_RATE'])


# In[22]:


with tf.Session() as sess:
    sess.run(init_variables)
    #plt.plot(audio.eval(session=sess), label='real')
    plt.plot(gen_to_wav[0].eval(session=sess), label='generated')
plt.plot(np.int32([np.sin(x/1000)*16000+32256 for x in range(gen_to_wav.shape[1])]))


# In[23]:


with tf.Session() as sess:
    sess.run(init_variables)
    sl.write_audio_not_one_hot(audio=gen_to_wav[0], filename='output_0.wav', session=sess, quantization_channels=quant)

