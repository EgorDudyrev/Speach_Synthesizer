
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


from scipy import sparse


# In[3]:


DIRS = sl.get_dirs(parent_dir=PARENT_DIR)
M_PARAMS = sl.get_model_params()


# In[4]:


quant = M_PARAMS['QUANTIZATION_CHANNELS']
nbits = int(np.log2(quant))


# # Build model

# # Get Data

# Всего файлов:
# * Обучение (cv-valid-train): 391552
# * cv-valid-dev: 8152
# * cv-valid-test: 7990

# # Develop models

# ## New sparsity

# In[39]:


def get_sparse_tensor(sparse_matrix, var, block_size=(16,1), session=None):
    b0, b1 = block_size
    k = tf.cast(var.shape[0]//b0//b1,'int32')
    n0 = sparse_matrix.shape[0]*b0
    n1 = sparse_matrix.shape[1]*b1
    
    tops = tf.math.top_k(tf.reshape(sparse_matrix,(-1,)), k=k)
    ids_flat = tf.cast(tops.indices, 'int64')
    ids = tf.transpose(tf.stack([ids_flat//(sparse_matrix.shape[1]), ids_flat%(sparse_matrix.shape[1])]))
    ids_flat_expand_start = ids[:,0]*n1*b0+ids[:,1]*b1
    ids_flat_expand = tf.reshape(tf.concat([
        tf.concat([ids_flat_expand_start+j for j in range(b1)],0)
            +i*n1 for i in range(b0)],0), (-1,))
    ids_flat_expand = tf.contrib.framework.sort(ids_flat_expand)
    ids_expand = tf.transpose(tf.stack([ids_flat_expand//n1, ids_flat_expand%n1]))
    
    spv = tf.SparseTensor(indices=ids_expand, values=var, dense_shape=(n0,n1))
    return spv


# In[148]:


def get_sparse_matrix(k, sparse_tensor, block_size=(16,1)):
    b0,b1 = block_size
    spvd = tf.sparse.to_dense(sparse_tensor)
    shorted = tf.concat([[tf.split(t, spvd.shape[1]//b1, axis=1)] for t in tf.split(spvd, spvd.shape[0]//b0, axis=0)],0)
    shorted = tf.reshape(shorted, (-1,b0,b1))
    means = tf.reduce_mean(tf.abs(tf.reshape(shorted,(-1,b0*b1))),1)
    
    tops = tf.math.top_k(means, k=k)
    ids_flat = tops.indices
    ids = tf.transpose(tf.stack([ids_flat//(n1//b1), ids_flat%(n1//b1)]))
    M1 = tf.sparse.to_dense(tf.SparseTensor(indices=tf.cast(ids,'int64'),
                values=np.ones(shape=ids.shape[0]).astype('int64'),
                dense_shape=(n0//b0, n1//b1)), validate_indices=False)
    
    
    ids = tf.transpose(tf.stack([ids_flat//(n1//b1), ids_flat%(n1//b1)]))
    ids_flat_expand_start = ids[:,0]*n1*b0+ids[:,1]*b1
    ids_flat_expand = tf.reshape(tf.concat([
        tf.concat([ids_flat_expand_start+i*n1 for i in range(b0)],0)
            +j for j in range(b1)],0), (-1,))
    ids_flat_expand = tf.contrib.framework.sort(ids_flat_expand)
    vals = tf.gather(tf.reshape(spvd, (-1,)), ids_flat_expand)
    return M1, vals


# In[149]:


b0, b1 = 16, 1


# In[248]:


n0, n1 = 64, 5


# In[249]:


tf.reset_default_graph()
sess = tf.Session()


# Задание переменных и матрицы разреженности

# In[250]:


ids = np.concatenate([[(i,j) for j in range(n1//b1)] for i in range(n0//b0)])
M   = tf.sparse.to_dense(tf.SparseTensor(indices=tf.cast(ids,'int64'),
                values=np.ones(shape=ids.shape[0]).astype('int64'),
                dense_shape=(n0//b0, n1//b1)), validate_indices=False)


# In[251]:


var = tf.Variable(tf.truncated_normal(shape=(n0*n1,), mean=0, stddev=0.1))


# In[252]:


sess.run(tf.global_variables_initializer())


# Из матрицы разреженности и переменных получаем матрицу переменных

# In[253]:


sess.run(M)


# In[254]:


sess.run(var)


# In[255]:


spv = get_sparse_tensor(M, var, block_size=(b0,b1), session=sess)


# In[256]:


tf.sparse.to_dense(spv).eval(session=sess)


# In[257]:


var


# Прореживаем матрицу и вектор переменных

# sparsity_level = 0.5 => k = M.size*0.5

# In[258]:


M1, vals = get_sparse_matrix(10, spv, block_size=(b0,b1))


# In[259]:


M1.eval(session=sess)


# In[260]:


vals = vals.eval(session=sess)


# In[261]:


var1 = tf.Variable(vals)
sess.run(var1.initializer)


# In[262]:


spv1 = get_sparse_tensor(M1, var1, block_size=(b0,b1))


# Прореживаем ещё раз

# sparsity_level = 0.9 => k = M.size*0.1

# In[263]:


M2, vals = get_sparse_matrix(2, spv, block_size=(b0,b1))


# In[264]:


M2.eval(session=sess)


# In[265]:


vals = sess.run(vals)
var2 = tf.Variable(vals)
sess.run(var2.initializer)


# In[266]:


spv2 = get_sparse_tensor(M2, var2, block_size=(b0,b1))


# In[267]:


plt.figure(figsize=(20,9))
plt.subplot(131)
sns.heatmap(np.abs(sess.run(tf.sparse.to_dense(spv))), cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.3)

plt.subplot(132)
sns.heatmap(np.abs(sess.run(tf.sparse.to_dense(spv1))), cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.3)

plt.subplot(133)
sns.heatmap(np.abs(sess.run(tf.sparse.to_dense(spv2))), cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.3)

plt.tight_layout()
plt.show()


# # New model with new loss

# In[146]:


import DWave4
DWave4 = importlib.reload(DWave4)


# In[147]:


tf.reset_default_graph()
sess = tf.Session()


# In[148]:


gru = DWave4.WaveGRU(3,32, n_batches=1, hidden_size=256)
sess.run(tf.global_variables_initializer())


# In[149]:


gru.train(audio_data, txt_emb, sess)


# In[137]:


txt_emb = np.array([0]*gru.text_embed_size).reshape(1,-1)


# In[9]:


aud = gru.generate_audio(txt_emb, sess, seconds=1, show_tqdm=True);


# In[13]:


fname = DIRS['RAW_DATA']+'/rus/voxforge_ru/0/00/78d77cdb75be'


# In[14]:


os.path.isfile(fname+'.wav')


# In[120]:


audio_data = sl.load_data([fname+'.wav']).eval(session=sess)
audio_data


# In[127]:


audio_data.shape

audio_data = np.concatenate([audio_data,np.ones((1,30-1-int(audio_data.shape[1]%30)+30,2))],1)
audio_data = np.concatenate([-np.ones((1,30*2,2)),audio_data],1)n_batches=30for i in range(n_batches-1):
    audio_data = np.concatenate([audio_data[:,1:,:],audio_data[:,:-1,-2:]],2)audio_data.shapeaudio_data_resh = audio_data.reshape((1,30,-1,2))audio_data_resh.shape
# # New model with new sparse

# In[27]:


import DWave5
DWave5 = importlib.reload(DWave5)


# In[28]:


tf.reset_default_graph()
sess = tf.Session()


# In[29]:


gru = DWave5.WaveGRU(3,32, n_batches=1)
sess.run(tf.global_variables_initializer())


# In[30]:


txt_emb = np.array([0]*gru.text_embed_size).reshape(1,-1)


# In[31]:


gru.generate(txt_emb, sess, seconds=1, show_tqdm=True);


# In[32]:


gru.sparsify(10, sess)


# In[33]:


sns.heatmap(tf.sparse.to_dense(gru.O2).eval(session=sess))


# In[20]:


get_ipython().run_line_magic('timeit', 'sess.run(tf.sparse.to_dense(gru.O1))')


# In[21]:


get_ipython().run_line_magic('timeit', 'sess.run(tf.transpose(tf.sparse.to_dense(gru.O1)))')


# In[22]:


tf.sparse_matmul


# In[632]:


fname = DIRS['RAW_DATA']+'/rus/voxforge_ru/0/00/78d77cdb75be'


# In[34]:


gru.generate(txt_emb, sess, seconds=1, show_tqdm=True);


# In[650]:


st = gru.O1
dt = tf.sparse.to_dense(gru.O1)
b = tf.sparse.to_dense(gru.O3)


# In[719]:


ddt = tf.truncated_normal(shape=(112,112))


# In[722]:


get_ipython().run_line_magic('timeit', 'tf.matmul(ddt,b).eval(session=sess)')


# In[715]:


get_ipython().run_line_magic('timeit', 'tf.matmul(dt,b).eval(session=sess)')


# In[714]:


get_ipython().run_line_magic('timeit', 'tf.sparse_matmul(dt, b, a_is_sparse=True).eval(session=sess)')


# In[713]:


get_ipython().run_line_magic('timeit', 'tf.sparse.matmul(st, b).eval(session=sess)')


# In[678]:


get_ipython().run_line_magic('timeit', 'tf.sparse.to_dense(st).eval(session=sess)')


# In[679]:


get_ipython().run_line_magic('timeit', 'tf.matmul(dt, b).eval(session=sess)')


# In[680]:


from scipy import sparse


# In[685]:


sm = sparse.rand(100,100)


# In[687]:


dm = sm.todense()


# In[692]:


bm = np.random.uniform(1000, size=(100,50))


# In[694]:


bm.shape


# In[696]:


sm.dot(bm)


# In[703]:


get_ipython().run_line_magic('timeit', 'sm.dot(bm)')


# In[704]:


get_ipython().run_line_magic('timeit', 'dm.dot(bm)')


# In[666]:


tf.linalg.matmul(st,b, a_is_sparse=True)


# In[662]:


sns.heatmap(dt.eval(session=sess))


# In[643]:


st


# In[644]:


dt

