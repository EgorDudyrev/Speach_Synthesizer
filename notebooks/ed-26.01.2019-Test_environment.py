
# coding: utf-8

# In[18]:


import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import seaborn as sns


# In[21]:


sys.version


# In[22]:


libs = [np, pd, tf, matplotlib, sns]
for lib in libs:
    print(f'{lib.__name__}: {lib.__version__}')

