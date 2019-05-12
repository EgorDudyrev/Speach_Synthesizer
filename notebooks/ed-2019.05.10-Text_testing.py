
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
import re
import scipy.signal as sgn

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


from pathlib import Path
import json


# In[3]:


DIRS = sl.get_dirs(parent_dir=PARENT_DIR)
M_PARAMS = sl.get_model_params()


# In[4]:


quant = M_PARAMS['QUANTIZATION_CHANNELS']
nbits = int(np.log2(quant))


# # Voxforge text

# In[5]:


text_ds = pd.read_csv(DIRS['RAW_DATA']+'rus/voxforge_ru.csv', names=['wav','txt','dur'])
print(text_ds.shape)
text_ds.head()


# In[6]:


text_ds['wav'] =  text_ds['wav'].apply(lambda x: x.replace('../data/ru_open_stt', DIRS['RAW_DATA']+'rus'))
text_ds['txt'] =  text_ds['txt'].apply(lambda x: x.replace('../data/ru_open_stt', DIRS['RAW_DATA']+'rus'))


# In[15]:


fname_txt, fname_wav = text_ds.loc[0,['txt','wav']]

with open(fname_txt, 'rb') as f:
    txtb = f.read().strip()#.decode()
txt = txtb.decode()

tf.reset_default_graph()
audio = sl.load_audio_not_one_hot(fname_wav)
with tf.Session() as sess:
    init_variables = tf.global_variables_initializer()
    sess.run(init_variables)
    audio_ev = audio.eval(session=sess)

audio_ev_norm, audio_roll, lim, m = sl.align_audio(audio_ev, txt, window_size=1000)


# In[17]:


sl = importlib.reload(sl)


# In[18]:


for idx, row in tqdm_notebook(text_ds.iterrows(), total=len(text_ds)):
    fname_txt, fname_wav = row['txt'], row['wav']
    
    with open(fname_txt, 'rb') as f:
        txtb = f.read().strip()#.decode()
    txt = txtb.decode()

    tf.reset_default_graph()
    audio = sl.load_audio_not_one_hot(fname_wav)
    with tf.Session() as sess:
        init_variables = tf.global_variables_initializer()
        sess.run(init_variables)
        audio_ev = audio.eval(session=sess)

    audio_ev_norm, audio_roll, lim, m = sl.align_audio(audio_ev, txt, window_size=1000)
    if m is None:
        fname_align = None
    else:
        fname_align = fname_txt.replace('txt','align')
        with open(fname_align, 'wb') as f:
            np.save(f,m)
    text_ds.loc[idx,'align'] = fname_align


# In[19]:


text_ds.to_csv(DIRS['RAW_DATA']+'rus/voxforge_ru_aligned.csv')


# In[20]:


plt.figure(figsize=(20,10))
sl.plot_audio(audio_ev_norm, audio_roll, lim, m, ' '.join([c.replace(' ','_') for c in m[:,0]]))


# # Maximum text characteristics

# In[1586]:


chars = set()
max_text_len = 0
max_word_len = 0
max_words_in_text = 0
for fname in tqdm_notebook(txt_fnames):
    with open(fname, 'rb') as f:
        t = f.read().decode().strip()
    [chars.add(x) for x in t];
    max_text_len = max(max_text_len, len(t))
    max_word_len = max(max_word_len,max([len(x) for x in t.split(' ')]))
    max_words_in_text = max(max_words_in_text,len(t.split(' ')))


# In[1587]:


max_text_len, max_word_len, max_words_in_text, len(chars)


# In[86]:


max_text_len = 210
max_word_len = 35 #20
max_words_in_text = 33


# In[1588]:


211*35


# # Add text to Input features

# In[21]:


char_to_int = {k: idx for idx,k in enumerate('\0 абвгдежзийклмнопрстуфхцчшщъыьэюяё')}


# In[70]:


fname_wav, fname_txt, fname_align = text_ds.loc[0,['wav','txt','align']]
fname_wav


# In[71]:


data = sl.load_data([fname_wav])


# In[72]:


oh = sl.load_text_oh([fname_align])


# In[73]:


sl.get_train_test(data, batch_size=10, truncated_len=100, text_oh=oh)


# In[65]:


data


# In[66]:


oh


# In[57]:


sl.generate_batch(data, 10, 100, text_oh=)

