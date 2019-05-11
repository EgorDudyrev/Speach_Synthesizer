
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


def train_model(model_class,input_dimensions, hidden_size, batch_size, truncated_len, num_epochs, model_name,
print_period=50, save_period=50, log_period=50, n_files_per_epoch=5, sparsify_epochs = [], sparsity_level=1):
    if model_name not in os.listdir(DIRS['MODELS']):
        os.mkdir(DIRS['MODELS']+model_name)
    
    tf.reset_default_graph()
    model = model_class(input_dimensions, hidden_size)
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
                X = sl.load_data(wav_fnames, n_files_per_epoch)
                total_series_length = int(X.shape[1])
                epochs_per_files_last = total_series_length//batch_size//truncated_len
            epochs_per_files_last-=1
            
            if epoch in sparsify_epochs:
                k = model.calc_sparsity_level(epoch, sparsify_epochs, sparsity_level)
                model.sparsify(k, sess)
            
            X_train, Y_train, X_test, Y_test = sl.get_train_test(X, batch_size, truncated_len, sess)
            train_loss, validation_loss = model.train(X_train, Y_train, X_test, Y_test, sess)

            # Log the losses
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            
            msg = f'Iteration: {epoch}, train loss: {train_loss:.4f}, val loss: {validation_loss:.4f}'
            # Display an update every 50 iterations
            if epoch % print_period == 0 and epoch!=0:
                print(msg)
            if epoch % print_period == 0 and epoch!=0:
                sl.plot_losses(train_losses, validation_losses, title=msg)
                plt.show()
            if epoch % save_period == 0:
                saver.save(sess, DIRS['MODELS']+model_name+'/checkpoint',global_step=epoch,write_meta_graph=True)
        
        sl.plot_losses(train_losses, validation_losses, title=msg)
        plt.show()

        saver.save(sess, DIRS['MODELS']+model_name+'/final')
        
    return train_losses, validation_losses, model


# # Get Data

# Всего файлов:
# * Обучение (cv-valid-train): 391552
# * cv-valid-dev: 8152
# * cv-valid-test: 7990

# # Develop models

# ## Simple Dense model

# In[5]:


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
        self.Ir = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wr')
        self.Iu = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wu')
        self.Ie = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='We')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Rr = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ur')
        self.Ru = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uu')
        self.Re = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ue')
        
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
        self.Ir = tf.Variable(variables['Ir:0'], name='Ir')
        self.Iu = tf.Variable(variables['Iu:0'], name='Iu')
        self.Ie = tf.Variable(variables['Ie:0'], name='Ie')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Rr = tf.Variable(variables['Rr:0'], name='Rr')
        self.Ru = tf.Variable(variables['Ru:0'], name='Ru')
        self.Re = tf.Variable(variables['Re:0'], name='Re')
        
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
        
        self.y_c, self.y_f = tf.split(self.h_t_transposed, num_or_size_splits=2, axis=2, name='split_y')
        # Transpose the result back
        #self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')
        
        self.P_ct = tf.scan(self.get_P_cs, self.y_c, name='calc_Pc')
        self.c_t_transposed = tf.reduce_max(self.P_ct, axis=2, name='c_t_trans')
        self.c_t = tf.transpose(self.c_t_transposed, name='c_t')
        self.P_ft = tf.scan(self.get_P_fs, self.y_f, name='calc_Pf')
        self.f_t_transposed = tf.reduce_max(self.P_ft, axis=2, name='f_t_trans')
        self.f_t = tf.transpose(self.f_t_transposed, name='f_t')
        
        self.y = tf.stack([self.c_t, self.f_t], axis=2, name='y')
    
    def define_train_variables(self):
        self.output = self.y
        self.expected_output = tf.placeholder(
            dtype=tf.float64, shape=(None, None, 2), name='expected_output'
            #(batch_size, truncated_len, 2), name='expected_output'
        )
        #self.loss = tf.reduce_sum(0.5 * tf.pow(self.output - self.expected_output, 2)) / float(batch_size)
        # mean(1/2 * (y-y_true)^2)
        self.loss = tf.reduce_mean(0.5 * tf.pow(self.output - self.expected_output, 2), name='loss')
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        
    def get_P_cs(self, lastP, y_c):
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_c, self.O1)), self.O2), axis=1, name='P_c')
    def get_P_fs(self, lastP, y_f):
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_f, self.O3)), self.O4), axis=1, name='P_f')
        
    def forward_pass(self, h_tm1, x_t):
        """Perform a forward pass.
        Arguments
        ---------
        h_tm1: np.matrix
            The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix
            The input vector.
        """
        Iu_masked = tf.multiply(self.Iu, self.M, name='Iu')
        Ir_masked = tf.multiply(self.Ir, self.M, name='Ir')
        Ie_masked = tf.multiply(self.Ie, self.M, name='Ie')
        # Definitions of z_t and r_t
        u_t = tf.sigmoid(tf.matmul(h_tm1, self.Ru) + tf.matmul(x_t, Iu_masked) + self.bu, name='u_t')
        r_t = tf.sigmoid(tf.matmul(h_tm1, self.Rr) + tf.matmul(x_t, Ir_masked) + self.br, name='r_t')
        # Definition of h~_t
        e_t = tf.tanh(tf.multiply(r_t, tf.matmul(h_tm1, self.Re))+tf.matmul(x_t, Ir_masked)+ self.be, name='e_t')
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


# In[6]:


input_dimensions = 3
hidden_size = 16
batch_size = 10
truncated_len = 100 


# In[53]:


model_name = f'Sparse_Develop'

train_losses, validation_losses, model = train_model(
    WaveGRU, input_dimensions, hidden_size, batch_size, truncated_len,
    num_epochs=100, model_name=model_name, print_period=50, log_period=10
)


# In[54]:


plt.figure(figsize=(10,3))
plt.plot(train_losses, label='Train')
plt.plot(validation_losses, label='Validation')
plt.title(model_name)
plt.show()


# ## Simple Sparse Matrix
class WaveGRU_simple_sparse:
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
        
        self.M_Ir, self.M_Iu, self.M_Ie = [np.ones(shape=(self.input_dimensions, self.hidden_size)) for i in range(3)]
        self.M_Rr, self.M_Ru, self.M_Re = [np.ones(shape=(self.hidden_size, self.hidden_size)) for i in range(3)]
        self.M_O1, self.M_O2, self.M_O3, self.M_O4 = [np.ones(shape=(self.hidden_size//2, self.hidden_size//2)) for i in range(4)]
                
        
    def define_variables(self, dtype):     
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Ir = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Ir')
        self.Iu = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Iu')
        self.Ie = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Ie')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Rr = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Rr')
        self.Ru = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ru')
        self.Re = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Re')
        
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
        self.Ir = tf.Variable(variables['Ir:0'], name='Ir')
        self.Iu = tf.Variable(variables['Iu:0'], name='Iu')
        self.Ie = tf.Variable(variables['Ie:0'], name='Ie')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Rr = tf.Variable(variables['Rr:0'], name='Rr')
        self.Ru = tf.Variable(variables['Ru:0'], name='Ru')
        self.Re = tf.Variable(variables['Re:0'], name='Re')
        
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
        
        self.y_c, self.y_f = tf.split(self.h_t_transposed, num_or_size_splits=2, axis=2, name='split_y')
        # Transpose the result back
        #self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')
        
        self.P_ct = tf.scan(self.get_P_cs, self.y_c, name='calc_Pc')
        self.c_t_transposed = tf.reduce_max(self.P_ct, axis=2, name='c_t_trans')
        self.c_t = tf.transpose(self.c_t_transposed, name='c_t')
        self.P_ft = tf.scan(self.get_P_fs, self.y_f, name='calc_Pf')
        self.f_t_transposed = tf.reduce_max(self.P_ft, axis=2, name='f_t_trans')
        self.f_t = tf.transpose(self.f_t_transposed, name='f_t')
        
        self.y = tf.stack([self.c_t, self.f_t], axis=2, name='y')
    
    def define_train_variables(self):
        self.output = self.y
        self.expected_output = tf.placeholder(
            dtype=tf.float64, shape=(None, None, 2), name='expected_output'
            #(batch_size, truncated_len, 2), name='expected_output'
        )
        #self.loss = tf.reduce_sum(0.5 * tf.pow(self.output - self.expected_output, 2)) / float(batch_size)
        # mean(1/2 * (y-y_true)^2)
        self.loss = tf.reduce_mean(0.5 * tf.pow(self.output - self.expected_output, 2), name='loss')
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        
    def get_P_cs(self, lastP, y_c):
        O1_sparse, O2_sparse = tf.multiply(self.O1, self.M_O1, name='O1_sparse'), tf.multiply(self.O2, self.M_O2, name='O2_sparse')
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_c, O1_sparse)), O2_sparse), axis=1, name='P_c')
    def get_P_fs(self, lastP, y_f):
        O3_sparse, O4_sparse = tf.multiply(self.O3, self.M_O3, name='O3_sparse'), tf.multiply(self.O4, self.M_O4, name='O4_sparse')
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_f, O3_sparse)), O4_sparse), axis=1, name='P_f')
        
    def forward_pass(self, h_tm1, x_t):
        """Perform a forward pass.
        Arguments
        ---------
        h_tm1: np.matrix
            The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix
            The input vector.
        """
        self.Iu_masked = tf.multiply(self.Iu, self.M, name='Iu_mask')
        self.Ir_masked = tf.multiply(self.Ir, self.M, name='Ir_mask')
        self.Ie_masked = tf.multiply(self.Ie, self.M, name='Ie_mask')
        
        self.Iu_sparse = tf.multiply(self.Iu_masked, self.M_Iu, name='Iu_sparse')
        self.Ir_sparse = tf.multiply(self.Ir_masked, self.M_Ir, name='Ir_sparse')
        self.Ie_sparse = tf.multiply(self.Ie_masked, self.M_Ie, name='Ie_sparse')
        self.Ru_sparse = tf.multiply(self.Ru, self.M_Ru, name='Ru_sparse')
        self.Rr_sparse = tf.multiply(self.Rr, self.M_Rr, name='Rr_sparse')
        self.Re_sparse = tf.multiply(self.Re, self.M_Re, name='Re_sparse')
        
        
        # Definitions of z_t and r_t
        u_t = tf.sigmoid(tf.matmul(h_tm1, self.Ru_sparse) + tf.matmul(x_t, self.Iu_sparse) + self.bu, name='u_t')
        r_t = tf.sigmoid(tf.matmul(h_tm1, self.Rr_sparse) + tf.matmul(x_t, self.Ir_sparse) + self.br, name='r_t')
        # Definition of h~_t
        e_t = tf.tanh(tf.multiply(r_t, tf.matmul(h_tm1, self.Re_sparse))+tf.matmul(x_t, self.Ir_sparse)+ self.be, name='e_t')
        # Compute the next hidden state
        h_t = tf.multiply(u_t, h_tm1) + tf.multiply(1 - u_t, e_t)
        return h_t
    
    def get_sparse_matrix(self, tensor, k, session):
        flat = tf.reshape(tensor, (-1,))
        k_ = int(k*int(flat.shape[0]))
        idxs = session.run(tf.math.top_k(-flat, k_).indices)
        M = np.ones((int(flat.shape[0])))
        M[idxs] = 0
        return M.reshape(tensor.shape)
    
    def get_sparse_tensor(self, tensor, k, session):
        flat = tf.reshape(tensor, (-1,))
        nrows = int(tensor.shape[0])
        k_biggest = int(k*int(flat.shape[0]))
        idxs = session.run(tf.math.top_k(flat, k_biggest).indices)
        
        S
        M = np.ones((int(flat.shape[0])))
        M[idxs] = 0
        return M.reshape(tensor.shape)
    
    def sparsify(self, k, session):
        if self.use_sparse_tensor:
            pass
        else:
            self.M_Iu, self.M_Ir, self.M_Ie = [
                self.get_sparse_matrix(t, k, session) for t in [self.Iu, self.Ir, self.Ie]]
            self.M_Ru, self.M_Rr, self.M_Re = [
                self.get_sparse_matrix(t, k, session) for t in [self.Ru, self.Rr, self.Re]]
            self.M_O1, self.M_O2, self.M_O3, self.M_O4 = [
                self.get_sparse_matrix(t, k, session) for t in [self.M_O1, self.M_O2, self.M_O3, self.M_O4]]

    @staticmethod
    def calc_sparsity_level(t,sparsify_epochs ,sparsity_level):
        t0 = min(sparsify_epochs)
        S = max(sparsify_epochs)-t0
        Z = sparsity_level
        return Z*(1-(1-(t-t0)/S)**3)
    
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
# In[7]:


class WaveGRU_simple_sparse:
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
    
    def __init__(self, input_dimensions, hidden_size, dtype=tf.float64, variables_values_dict=None, use_sparse_tensor=False):
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        self.use_sparse_tensor = use_sparse_tensor
        
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
        
        if not self.use_sparse_tensor:
            self.M_Ir, self.M_Iu, self.M_Ie = [np.ones(shape=(self.input_dimensions, self.hidden_size)) for i in range(3)]
            self.M_Rr, self.M_Ru, self.M_Re = [np.ones(shape=(self.hidden_size, self.hidden_size)) for i in range(3)]
            self.M_O1, self.M_O2, self.M_O3, self.M_O4 = [np.ones(shape=(self.hidden_size//2, self.hidden_size//2)) for i in range(4)]
                
        
    def define_variables(self, dtype):     
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Ir, self.Iu, self.Ie = [
            tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01)
            for i in range(3)]
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Rr, self.Ru, self.Re = [
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01)
            for i in range(3)]
        # Biases for hidden vectors of shape (hidden_size,)
        self.br, self.bu, self.be = [
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01)
            for i in range(3)]
        # O's matrices
        self.O1, self.O2, self.O3, self.O4 = [
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size//2,self.hidden_size//2), mean=0, stddev=0.01)
            for i in range(4)]
                
        if self.use_sparse_tensor:
            def to_sparse(t):
                idx = tf.where(tf.not_equal(t, 0))
                return tf.SparseTensor(idx, tf.gather_nd(t, idx), t.get_shape())
            
            self.Ir, self.Iu, self.Ie, self.Rr, self.Ru, self.Re, self.O1, self.O2, self.O3, self.O4 = [
                #tf.contrib.layers.dense_to_sparse(t)
                to_sparse(t)
                for t in [self.Ir, self.Iu, self.Ie, self.Rr, self.Ru, self.Re, self.O1, self.O2, self.O3, self.O4]
            ]
            
            
        self.Ir,self.Iu, self.Ie =             tf.Variable(self.Ir, name='Ir'), tf.Variable(self.Iu, name='Iu'), tf.Variable(self.Ie, name='Ie')
        self.Rr,self.Ru, self.Re =             tf.Variable(self.Rr, name='Rr'), tf.Variable(self.Ru, name='Ru'), tf.Variable(self.Re, name='Re')
        self.br,self.bu, self.be =             tf.Variable(self.br, name='br'), tf.Variable(self.bu, name='bu'), tf.Variable(self.be, name='be')
        self.O1,self.O2, self.O3, self.O4 =             tf.Variable(self.O1, name='O1'), tf.Variable(self.O2, name='O2'), tf.Variable(self.O3, name='O3'), tf.Variable(self.O4, name='O4')
    
    def restore_variables(self, variables):
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Ir = tf.Variable(variables['Ir:0'], name='Ir')
        self.Iu = tf.Variable(variables['Iu:0'], name='Iu')
        self.Ie = tf.Variable(variables['Ie:0'], name='Ie')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Rr = tf.Variable(variables['Rr:0'], name='Rr')
        self.Ru = tf.Variable(variables['Ru:0'], name='Ru')
        self.Re = tf.Variable(variables['Re:0'], name='Re')
        
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
        
        self.y_c, self.y_f = tf.split(self.h_t_transposed, num_or_size_splits=2, axis=2, name='split_y')
        # Transpose the result back
        #self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')
        
        self.P_ct = tf.scan(self.get_P_cs, self.y_c, name='calc_Pc')
        self.c_t_transposed = tf.reduce_max(self.P_ct, axis=2, name='c_t_trans')
        self.c_t = tf.transpose(self.c_t_transposed, name='c_t')
        self.P_ft = tf.scan(self.get_P_fs, self.y_f, name='calc_Pf')
        self.f_t_transposed = tf.reduce_max(self.P_ft, axis=2, name='f_t_trans')
        self.f_t = tf.transpose(self.f_t_transposed, name='f_t')
        
        self.y = tf.stack([self.c_t, self.f_t], axis=2, name='y')
    
    def define_train_variables(self):
        self.output = self.y
        self.expected_output = tf.placeholder(
            dtype=tf.float64, shape=(None, None, 2), name='expected_output'
            #(batch_size, truncated_len, 2), name='expected_output'
        )
        #self.loss = tf.reduce_sum(0.5 * tf.pow(self.output - self.expected_output, 2)) / float(batch_size)
        # mean(1/2 * (y-y_true)^2)
        self.loss = tf.reduce_mean(0.5 * tf.pow(self.output - self.expected_output, 2), name='loss')
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        
    def get_P_cs(self, lastP, y_c):
        if not self.use_sparse_tensor:
            O1_sparse, O2_sparse = tf.multiply(self.O1, self.M_O1, name='O1_sparse'), tf.multiply(self.O2, self.M_O2, name='O2_sparse')
        else:
            O1_sparse, O2_sparse = self.O1, self.O2
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_c, O1_sparse)), O2_sparse), axis=1, name='P_c')
    def get_P_fs(self, lastP, y_f):
        if not self.use_sparse_tensor:
            O3_sparse, O4_sparse = tf.multiply(self.O3, self.M_O3, name='O3_sparse'), tf.multiply(self.O4, self.M_O4, name='O4_sparse')
        else:
            O3_sparse, O4_sparse = self.O4, self.O4
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_f, O3_sparse)), O4_sparse), axis=1, name='P_f')
        
    def forward_pass(self, h_tm1, x_t):
        """Perform a forward pass.
        Arguments
        ---------
        h_tm1: np.matrix
            The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix
            The input vector.
        """
        self.Iu_masked = tf.multiply(self.Iu, self.M, name='Iu_mask')
        self.Ir_masked = tf.multiply(self.Ir, self.M, name='Ir_mask')
        self.Ie_masked = tf.multiply(self.Ie, self.M, name='Ie_mask')
        
        if not self.use_sparse_tensor:
            self.Iu_sparse = tf.multiply(self.Iu_masked, self.M_Iu, name='Iu_sparse')
            self.Ir_sparse = tf.multiply(self.Ir_masked, self.M_Ir, name='Ir_sparse')
            self.Ie_sparse = tf.multiply(self.Ie_masked, self.M_Ie, name='Ie_sparse')
            self.Ru_sparse = tf.multiply(self.Ru, self.M_Ru, name='Ru_sparse')
            self.Rr_sparse = tf.multiply(self.Rr, self.M_Rr, name='Rr_sparse')
            self.Re_sparse = tf.multiply(self.Re, self.M_Re, name='Re_sparse')
        else:
            self.Iu_sparse, self.Ir_sparse, self.Ie_sparse = self.Iu, self.Ir, self.Ie
            self.Ru_sparse, self.Rr_sparse, self.Re_sparse = self.Ru, self.Rr, self.Re
        
        # Definitions of z_t and r_t
        u_t = tf.sigmoid(tf.matmul(h_tm1, self.Ru_sparse) + tf.matmul(x_t, self.Iu_sparse) + self.bu, name='u_t')
        r_t = tf.sigmoid(tf.matmul(h_tm1, self.Rr_sparse) + tf.matmul(x_t, self.Ir_sparse) + self.br, name='r_t')
        # Definition of h~_t
        e_t = tf.tanh(tf.multiply(r_t, tf.matmul(h_tm1, self.Re_sparse))+tf.matmul(x_t, self.Ir_sparse)+ self.be, name='e_t')
        # Compute the next hidden state
        h_t = tf.multiply(u_t, h_tm1) + tf.multiply(1 - u_t, e_t)
        return h_t
    
    def get_sparse_matrix(self, tensor, k, session):
        flat = tf.reshape(tensor, (-1,))
        k_ = int(k*int(flat.shape[0]))
        idxs = session.run(tf.math.top_k(-flat, k_).indices)
        M = np.ones((int(flat.shape[0])))
        M[idxs] = 0
        return M.reshape(tensor.shape)
    
    def get_sparse_tensor(self, tensor, k, session):
        flat = tf.reshape(tensor.to_dense(), (-1,))
        nrows = int(tensor.shape[0])
        k_biggest = int((1-k)*int(flat.shape[0]))
        idxs = session.run(tf.math.top_k(flat, k_biggest).indices)
        flat_e = flat.eval(session=session)
        SpT = tf.sparse.SparseTensor(indices=list(zip(idxs//nrows,idxs%nrows)), values=flat_e[idxs],dense_shape=tensor.shape)
        return SpT
    
    def sparsify(self, k, session):
        if self.use_sparse_tensor:
            self.Iu = self.get_sparse_tensor(self.Iu, k, session)
            self.Ir = self.get_sparse_tensor(self.Ir, k, session)
            self.Ie = self.get_sparse_tensor(self.Ie, k, session)
            
            self.Ru = self.get_sparse_tensor(self.Ru, k, session)
            self.Rr = self.get_sparse_tensor(self.Rr, k, session)
            self.Re = self.get_sparse_tensor(self.Re, k, session)
            
            self.O1 = self.get_sparse_tensor(self.O1, k, session)
            self.O2 = self.get_sparse_tensor(self.O2, k, session)
            self.O3 = self.get_sparse_tensor(self.O3, k, session)
            self.O4 = self.get_sparse_tensor(self.O4, k, session)
        else:
            self.M_Iu, self.M_Ir, self.M_Ie = [
                self.get_sparse_matrix(t, k, session) for t in [self.Iu, self.Ir, self.Ie]]
            self.M_Ru, self.M_Rr, self.M_Re = [
                self.get_sparse_matrix(t, k, session) for t in [self.Ru, self.Rr, self.Re]]
            self.M_O1, self.M_O2, self.M_O3, self.M_O4 = [
                self.get_sparse_matrix(t, k, session) for t in [self.M_O1, self.M_O2, self.M_O3, self.M_O4]]

    @staticmethod
    def calc_sparsity_level(t,sparsify_epochs ,sparsity_level):
        t0 = min(sparsify_epochs)
        S = max(sparsify_epochs)-t0
        Z = sparsity_level
        return Z*(1-(1-(t-t0)/S)**3)
    
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


# In[8]:


input_dimensions = 3
hidden_size = 16
batch_size = 10
truncated_len = 100 

tf.reset_default_graph()
with tf.Session() as sess:
    #model = WaveGRU_simple_sparse(input_dimensions, hidden_size)
    sess.run(tf.global_variables_initializer())
    
    dtype = 'float32'
    m = tf.truncated_normal(dtype=dtype, shape=(3, 5), mean=0, stddev=0.01)
    m_e = m.eval(session=sess)
    
    def to_sparse(t):
        idx = tf.where(tf.not_equal(t, 0))
        return tf.SparseTensor(idx, tf.gather_nd(t, idx), t.get_shape())
    s = to_sparse(m)
    s_e = s.eval(session=sess)
    
    v = tf.Variable(s)
# In[192]:


model_name = f'Sparse_Develop'

train_losses, validation_losses, model = train_model(
    WaveGRU_simple_sparse, input_dimensions, hidden_size, batch_size=batch_size, truncated_len=truncated_len,
    num_epochs=50, model_name=model_name, print_period=50, log_period=10,
    sparsify_epochs = [10, 20, 30, 45], sparsity_level=0.95
)


# In[193]:


plt.figure(figsize=(10,3))
plt.plot(train_losses, label='Train')
plt.plot(validation_losses, label='Validation')
plt.title(model_name)
plt.show()


# ## Structured Sparse

# In[9]:


class WaveGRU_structured_sparse:
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
    
    def __init__(self, input_dimensions, hidden_size, dtype=tf.float64, variables_values_dict=None, block_shape=(16,1)):
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        self.block_shape = block_shape
        
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
        
        m1,m2 = self.block_shape
        self.M_Ir, self.M_Iu, self.M_Ie = [np.ones(shape=(self.input_dimensions, self.hidden_size)) for i in range(3)]
        self.M_Rr, self.M_Ru, self.M_Re = [np.ones(shape=(self.hidden_size//m1, self.hidden_size//m2)) for i in range(3)]
        self.M_O1, self.M_O2, self.M_O3, self.M_O4 = [np.ones(shape=(self.hidden_size//2//m1, self.hidden_size//2//m2)) for i in range(4)]
                
        
    def define_variables(self, dtype):     
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Ir = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Ir')
        self.Iu = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Iu')
        self.Ie = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Ie')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Rr = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Rr')
        self.Ru = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ru')
        self.Re = tf.Variable(tf.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Re')
        
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
        self.Ir = tf.Variable(variables['Ir:0'], name='Ir')
        self.Iu = tf.Variable(variables['Iu:0'], name='Iu')
        self.Ie = tf.Variable(variables['Ie:0'], name='Ie')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Rr = tf.Variable(variables['Rr:0'], name='Rr')
        self.Ru = tf.Variable(variables['Ru:0'], name='Ru')
        self.Re = tf.Variable(variables['Re:0'], name='Re')
        
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
        
        self.y_c, self.y_f = tf.split(self.h_t_transposed, num_or_size_splits=2, axis=2, name='split_y')
        # Transpose the result back
        #self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')
        
        self.P_ct = tf.scan(self.get_P_cs, self.y_c, name='calc_Pc')
        self.c_t_transposed = tf.reduce_max(self.P_ct, axis=2, name='c_t_trans')
        self.c_t = tf.transpose(self.c_t_transposed, name='c_t')
        self.P_ft = tf.scan(self.get_P_fs, self.y_f, name='calc_Pf')
        self.f_t_transposed = tf.reduce_max(self.P_ft, axis=2, name='f_t_trans')
        self.f_t = tf.transpose(self.f_t_transposed, name='f_t')
        
        self.y = tf.stack([self.c_t, self.f_t], axis=2, name='y')
    
    def define_train_variables(self):
        self.output = self.y
        self.expected_output = tf.placeholder(
            dtype=tf.float64, shape=(None, None, 2), name='expected_output'
            #(batch_size, truncated_len, 2), name='expected_output'
        )
        #self.loss = tf.reduce_sum(0.5 * tf.pow(self.output - self.expected_output, 2)) / float(batch_size)
        # mean(1/2 * (y-y_true)^2)
        self.loss = tf.reduce_mean(0.5 * tf.pow(self.output - self.expected_output, 2), name='loss')
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        
    def get_P_cs(self, lastP, y_c):
        M_O1, M_O2 = [self.extend_matrix(M) for M in [self.M_O1, self.M_O2]]
        O1_sparse, O2_sparse = tf.multiply(self.O1, M_O1, name='O1_sparse'), tf.multiply(self.O2, M_O2, name='O2_sparse')
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_c, O1_sparse)), O2_sparse), axis=1, name='P_c')
    def get_P_fs(self, lastP, y_f):
        M_O3, M_O4 = [self.extend_matrix(M) for M in [self.M_O3, self.M_O4]]
        O3_sparse, O4_sparse = tf.multiply(self.O3, M_O3, name='O3_sparse'), tf.multiply(self.O4, M_O4, name='O4_sparse')
        return tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(y_f, O3_sparse)), O4_sparse), axis=1, name='P_f')
        
    def forward_pass(self, h_tm1, x_t):
        """Perform a forward pass.
        Arguments
        ---------
        h_tm1: np.matrix
            The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix
            The input vector.
        """
        self.Iu_masked = tf.multiply(self.Iu, self.M, name='Iu_mask')
        self.Ir_masked = tf.multiply(self.Ir, self.M, name='Ir_mask')
        self.Ie_masked = tf.multiply(self.Ie, self.M, name='Ie_mask')
                      
        #M_Iu, M_Ir, M_Ie = [self.extend_matrix(M) for M in [self.M_Iu, self.M_Ir, self.M_Ie]]
        M_Ru, M_Rr, M_Re = [self.extend_matrix(M) for M in [self.M_Ru, self.M_Rr, self.M_Re]]
                
        #self.Iu_sparse = tf.multiply(self.Iu_masked, M_Iu, name='Iu_sparse')
        #self.Ir_sparse = tf.multiply(self.Ir_masked, M_Ir, name='Ir_sparse')
        #self.Ie_sparse = tf.multiply(self.Ie_masked, M_Ie, name='Ie_sparse')
        self.Iu_sparse, self.Ir_sparse, self.Ie_sparse = self.Iu_masked, self.Ir_masked, self.Ie_masked
        self.Ru_sparse = tf.multiply(self.Ru, M_Ru, name='Ru_sparse')
        self.Rr_sparse = tf.multiply(self.Rr, M_Rr, name='Rr_sparse')
        self.Re_sparse = tf.multiply(self.Re, M_Re, name='Re_sparse')
        
        
        # Definitions of z_t and r_t
        u_t = tf.sigmoid(tf.matmul(h_tm1, self.Ru_sparse) + tf.matmul(x_t, self.Iu_sparse) + self.bu, name='u_t')
        r_t = tf.sigmoid(tf.matmul(h_tm1, self.Rr_sparse) + tf.matmul(x_t, self.Ir_sparse) + self.br, name='r_t')
        # Definition of h~_t
        e_t = tf.tanh(tf.multiply(r_t, tf.matmul(h_tm1, self.Re_sparse))+tf.matmul(x_t, self.Ir_sparse)+ self.be, name='e_t')
        # Compute the next hidden state
        h_t = tf.multiply(u_t, h_tm1) + tf.multiply(1 - u_t, e_t)
        return h_t
    
    def get_sparse_matrix(self, tensor, k, session):
        tensor_ev = tensor.eval(session=session)
        m1,m2 = self.block_shape
        shorted = np.array([[tensor_ev[m1*i:m1*(i+1),m2*j:m2*(j+1)].mean() for j in range(tensor_ev.shape[1]//m2)] for i in range(tensor_ev.shape[0]//m1)])
        flat = shorted.flatten()
        k_ = int(k*int(flat.shape[0]))
        idxs = flat.argsort()[:k_]
        M = np.ones((int(flat.shape[0])))
        M[idxs] = 0
        return M.reshape(shorted.shape)
        
        #flat = tf.reshape(tensor, (-1,))
        #k_ = int(k*int(flat.shape[0]))
        #idxs = session.run(tf.math.top_k(-flat, k_).indices)
        #M = np.ones((int(flat.shape[0])))
        #M[idxs] = 0
        #return M.reshape(tensor.shape)
    
    def sparsify(self, k, session):
        #self.M_Iu, self.M_Ir, self.M_Ie = [
        #    self.get_sparse_matrix(t, k, session) for t in [self.Iu, self.Ir, self.Ie]]
        self.M_Ru, self.M_Rr, self.M_Re = [
            self.get_sparse_matrix(t, k, session) for t in [self.Ru, self.Rr, self.Re]]
        self.M_O1, self.M_O2, self.M_O3, self.M_O4 = [
            self.get_sparse_matrix(t, k, session) for t in [self.O1, self.O2, self.O3, self.O4]]
            #self.get_sparse_matrix(t, k, session) for t in [self.M_O1, self.M_O2, self.M_O3, self.M_O4]]

    def extend_matrix(self, M):
        coefs = self.block_shape
        return np.concatenate([[np.concatenate([[j]*coefs[1] for j in i])]*coefs[0] for i in M])                
        
    @staticmethod
    def calc_sparsity_level(t,sparsify_epochs ,sparsity_level):
        t0 = min(sparsify_epochs)
        S = max(sparsify_epochs)-t0
        Z = sparsity_level
        return Z*(1-(1-(t-t0)/S)**3)
    
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


# In[10]:


input_dimensions = 3
hidden_size = 1024
batch_size = 10
truncated_len = 150 


# In[420]:


model_name = f'Sparse_Develop'

train_losses, validation_losses, model = train_model(
    WaveGRU_structured_sparse, input_dimensions, hidden_size, batch_size=batch_size, truncated_len=truncated_len,
    num_epochs=1000, model_name=model_name, print_period=50, log_period=10,
    sparsify_epochs = (np.array([0.5,0.6,0.7,0.8,0.9])*1000).astype(int),
    sparsity_level=0.95
)


# In[421]:


plt.figure(figsize=(10,3))
plt.plot(train_losses, label='Train')
plt.plot(validation_losses, label='Validation')
plt.title(model_name)
plt.show()


# # Restoring model

# In[11]:


model_name = f'Sparse_Develop'


# In[12]:


sorted(os.listdir(DIRS['MODELS']+model_name))


# In[13]:


tf.reset_default_graph()
#saver = tf.train.import_meta_graph(DIRS['MODELS']+model_name+'/final.meta')
saver = tf.train.import_meta_graph(DIRS['MODELS']+model_name+'/checkpoint-900.meta')
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint(DIRS['MODELS']+model_name))
    restored_variables = {x.name:x.eval(session=sess) for x in tf.global_variables()[:13]}

tf.reset_default_graph()
gru = WaveGRU(input_dimensions, hidden_size, variables_values_dict=restored_variables)

X = sl.load_data(wav_fnames, 3)
# # Sound generation

# In[15]:


with tf.Session() as sess:
    init_variables = tf.global_variables_initializer()
    sess.run(init_variables)
    gen_to_wav = gru.generate_sound(num_pieces=1, n_seconds=2, session=sess, sample_rate=M_PARAMS['SAMPLE_RATE'])


# In[16]:


with tf.Session() as sess:
    sess.run(init_variables)
    #plt.plot(audio.eval(session=sess), label='real')
    plt.plot(gen_to_wav[0].eval(session=sess), label='generated')
plt.plot(np.int32([np.sin(x/1000)*16000+32256 for x in range(gen_to_wav.shape[1])]))


# In[31]:


with tf.Session() as sess:
    init_variables = tf.global_variables_initializer()
    sess.run(init_variables)
    O1, O2, O3, O4 = sess.run([gru.O1, gru.O2, gru.O3, gru.O4])
    Iu, Ir, Ie = sess.run([gru.Iu, gru.Ir, gru.Ie])
    Ru, Rr, Re = sess.run([gru.Ru, gru.Rr, gru.Re])


# In[41]:


for idx, m in enumerate([O1, O2, O3, O4]):
    plt.subplot(1,4,idx+1)
    plt.imshow(m)
    plt.title(['O1','O2','O3','O4'][idx])
plt.tight_layout()
plt.show()

for idx, m in enumerate([Ru, Rr, Re]):
    plt.subplot(1,3,idx+1)
    plt.imshow(m)
    plt.title(['Ru','Rr','Re'][idx])
plt.tight_layout()
plt.show()

for idx, m in enumerate([Iu, Ir, Ie]):
    plt.subplot(1,3,idx+1)
    sns.heatmap(m)
    plt.title(['Iu','Ir','Ie'][idx])
plt.tight_layout()
plt.show()


# In[50]:


os.listdir(DIRS['RAW_DATA']+'cv_corpus_v1/cv-other-train')


# In[56]:


audio = sl.load_audio_not_one_hot(DIRS['RAW_DATA']+'cv_corpus_v1/cv-other-train/sample-052026.wav')


# In[52]:


X = sl.load_data([DIRS['RAW_DATA']+'cv_corpus_v1/cv-other-train/sample-052026.wav'])


# In[138]:


with tf.Session() as sess:
    init_variables = tf.global_variables_initializer()
    sess.run(init_variables)
    O1, O2, O3, O4 = sess.run([gru.O1, gru.O2, gru.O3, gru.O4])
    Iu, Ir, Ie = sess.run([gru.Iu, gru.Ir, gru.Ie])
    Ru, Rr, Re = sess.run([gru.Ru, gru.Rr, gru.Re])
    audio_eval = sess.run(audio)
    X_eval = sess.run(X)
    X_train, Y_train, X_test, Y_test = sl.get_train_test(X, 10, 5000)
    Y_train_audio = ((Y_train*128+128)[:,:,0])*256+(Y_train*128+128)[:,:,1]
    Y_train_audio_eval = sess.run(Y_train_audio)
    


# In[139]:


plt.plot(audio_eval, color='blue', label='audio')
for i in Y_train_audio_eval[:1]:
    plt.plot(i)
plt.legend()
plt.show()


# In[62]:


X_train, Y_train, X_test, Y_test = sl.get_train_test(X, 10, 100)
[x.shape for x in [X_train, Y_train, X_test, Y_test]]


# In[91]:


((Y_train*256+256)[:,:,0]*256)+(Y_train*256+256)[:,:,1]


# In[77]:


tf.concat(Y_train, axis=0)


# In[68]:


sl.mu_law_encode(Y_train, 256*256)


# In[63]:


Y_train

with tf.Session() as sess:
    sess.run(init_variables)
    sl.write_audio_not_one_hot(audio=gen_to_wav[0], filename='output_0.wav', session=sess, quantization_channels=quant)