
# coding: utf-8

# In[2]:


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


# In[3]:


DIRS = sl.get_dirs(parent_dir=PARENT_DIR)
M_PARAMS = sl.get_model_params()


# In[4]:


quant = M_PARAMS['QUANTIZATION_CHANNELS']
nbits = int(np.log2(quant))


# # Build model

# In[229]:


WaveGRU_simple_sparse.calc_sparsity_level


# In[234]:



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

# In[29]:


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


# In[30]:


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

# In[284]:


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
        self.Iu_masked = tf.multiply(self.Iu, self.M, name='Iu')
        self.Ir_masked = tf.multiply(self.Ir, self.M, name='Ir')
        self.Ie_masked = tf.multiply(self.Ie, self.M, name='Ie')
        
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
    
    def sparsify(self, k, session):
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


# In[272]:


input_dimensions = 3
hidden_size = 16
batch_size = 10
truncated_len = 100 


# In[ ]:


model_name = f'Sparse_Develop'

train_losses, validation_losses, model = train_model(
    WaveGRU_simple_sparse, input_dimensions, hidden_size, batch_size=batch_size, truncated_len=truncated_len,
    num_epochs=500, model_name=model_name, print_period=50, log_period=10,
    sparsify_epochs = [300, 350, 400, 450], sparsity_level=0.95
)


# In[ ]:


plt.figure(figsize=(10,3))
plt.plot(train_losses, label='Train')
plt.plot(validation_losses, label='Validation')
plt.title(model_name)
plt.show()


# # Restoring model
tf.reset_default_graph()
saver = tf.train.import_meta_graph(DIRS['MODELS']+model_name+'/final.meta')
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint(DIRS['MODELS']+model_name))
    restored_variables = {x.name:x.eval(session=sess) for x in tf.global_variables()[:13]}tf.reset_default_graph()
gru = WaveGRU(input_dimensions, hidden_size, variables_values_dict=restored_variables)X = sl.load_data(wav_fnames, 3)batch_size = 10
truncated_len = 100#M_PARAMS['SAMPLE_RATE']
total_series_length = int(X.shape[1])
num_epochs = 50#400#total_series_length//batch_size//truncated_len
print(batch_size, truncated_len, total_series_length, num_epochs)init_variables = tf.global_variables_initializer()with tf.Session() as sess:
    sess.run(init_variables)
    O1_restored = gru.O1.eval(session=sess)plt.figure(figsize=(15,4))
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
with tf.Session() as sess:
    sess.run(init_variables)
    gen_to_wav = gru.generate_sound(num_pieces=1, n_seconds=2, session=sess, sample_rate=M_PARAMS['SAMPLE_RATE'])with tf.Session() as sess:
    sess.run(init_variables)
    #plt.plot(audio.eval(session=sess), label='real')
    plt.plot(gen_to_wav[0].eval(session=sess), label='generated')
plt.plot(np.int32([np.sin(x/1000)*16000+32256 for x in range(gen_to_wav.shape[1])]))with tf.Session() as sess:
    sess.run(init_variables)
    sl.write_audio_not_one_hot(audio=gen_to_wav[0], filename='output_0.wav', session=sess, quantization_channels=quant)