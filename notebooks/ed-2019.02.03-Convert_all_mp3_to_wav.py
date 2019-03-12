
# coding: utf-8

# In[1]:


import os
import sys
import importlib
sys.path.append('/opt/notebooks/')
  
import numpy as np
from tqdm import tqdm_notebook
from joblib import Parallel, delayed
import subprocess
from pathlib import Path

try: importlib.reload(sl)
except: import synt_lib as sl


# In[2]:


DIRS = sl.get_dirs()
M_PARAMS = sl.get_model_params()


# # Get all mp3 files

# In[3]:


def convert_file(path, ext_to='.wav'):
    if isinstance(path, Path):
        path = path.as_posix()
    ext_from = '.'+path.split('.')[-1]
    newpath = path.replace(ext_from, ext_to)
    if os.path.isfile(newpath):
        return 2, newpath
    
    try:
        subprocess.run(['ffmpeg', '-loglevel', 'panic', '-i', path, '-ar', str(M_PARAMS['SAMPLE_RATE']), newpath])
        return 1, newpath
    except Exception as e:
        print(e)
        return 0, newpath


# Remove old wav files

# In[4]:


result = Path(DIRS['RAW_DATA']).rglob("*.mp3")
res_len = len(list(result))
result = Path(DIRS['RAW_DATA']).rglob("*.mp3")


# In[5]:


for fname in tqdm_notebook(result, total=res_len):
    newfname = fname.as_posix().replace('.mp3','.wav')
    if os.path.isfile(newfname):
        subprocess.run(['rm', newfname])


# Convert to new wav files

# In[6]:


result = Path(DIRS['RAW_DATA']).rglob("*.mp3")
res_len = len(list(result))
result = Path(DIRS['RAW_DATA']).rglob("*.mp3")


# In[7]:


res_len


# In[8]:


get_ipython().run_cell_magic('time', '', 'res_data = Parallel(n_jobs=4, verbose=4)(delayed(convert_file)(path) for path in result);')


# Кажется зависло. Запустим ещё раз

# In[9]:


result = Path(DIRS['RAW_DATA']).rglob("*.mp3")
res_len = len(list(result))
result = Path(DIRS['RAW_DATA']).rglob("*.mp3")


# In[11]:


result_wav = Path(DIRS['RAW_DATA']).rglob("*.wav")
res_len_wav = len(list(result_wav))
result_wav = Path(DIRS['RAW_DATA']).rglob("*.wav")


# In[19]:


res_len, res_len_wav


# А, нет. Всё отработало.
