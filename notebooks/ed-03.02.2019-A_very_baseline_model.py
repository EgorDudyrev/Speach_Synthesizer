
# coding: utf-8

# In[1]:


import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

import importlib
if '/opt/notebooks/' not in sys.path:
    sys.path.append('/opt/notebooks/')

try: importlib.reload(sl)
except: import synt_lib as sl


# In[2]:


DIRS = sl.get_dirs()
M_PARAMS = sl.get_model_params()


# # Get data for testing

# In[3]:


wav_fnames = Path(DIRS['RAW_DATA']).rglob("*.wav")
fname = wav_fnames.__next__().as_posix()


# In[4]:


X = sl.load_audio_one_hot(fname)


# In[5]:


sess = tf.Session()


# In[6]:


sl.write_audio_one_hot('sample.wav', X, sess)


# # Set model
