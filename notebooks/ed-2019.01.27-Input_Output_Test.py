
# coding: utf-8

# In[1]:


import os
import sys
import importlib
sys.path.append('/opt/notebooks/')

try: importlib.reload(sl)
except: import synt_lib as sl
    
import numpy as np


# In[2]:


DIRS = sl.get_dirs()
M_PARAMS = sl.get_model_params()


# # Получение .wav файлов

# .wav файлы

# In[3]:


[x for x in os.listdir(DIRS['SONGS'])[:5] if x.endswith('.wav')]


# .mp3 файлы

# In[4]:


[x for x in os.listdir(DIRS['SONGS'])[:5] if x.endswith('.mp3')]


# Преобразование .wav в .mp3

# In[5]:


for f in [x for x in os.listdir(DIRS['SONGS']) if x.endswith('.mp3')]:
    fname = os.path.join(DIRS['SONGS'], f)
    fname = fname.replace(' ', '\ ')
    get_ipython().system("ffmpeg -loglevel panic -i {fname} -ar {M_PARAMS['SAMPLE_RATE']} {fname.replace('.mp3','.wav')} -y")


# .wav файлы теперь

# In[6]:


[x for x in os.listdir(DIRS['SONGS'])[:5] if x.endswith('.wav')]


# # Работа с .wav файлами

# In[7]:


from wavenet import AudioReader, mu_law_encode, mu_law_decode
import librosa


# In[8]:


import tensorflow as tf


# In[9]:


def _one_hot(input_batch):
    '''One-hot encodes the waveform amplitudes.

    This allows the definition of the network as a categorical distribution
    over a finite set of possible amplitudes.
    '''
    with tf.name_scope('one_hot_encode'):
        encoded = tf.one_hot(
            input_batch,
            depth=M_PARAMS['QUANTISATION_CHANNELS'],
            dtype=tf.float32)
        shape = [M_PARAMS['BATCH_SIZE'], -1, M_PARAMS['QUANTISATION_CHANNELS']]
        encoded = tf.reshape(encoded, shape)
    return encoded


# In[10]:


def _de_one_hot(encoded):
    '''One-hot decodes the waveform amplitudes.
    '''
    with tf.name_scope('one_hot_decode'):
        decoded = tf.argmax(encoded, axis=2)
    return decoded


# In[11]:


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


# In[12]:


sess = tf.Session()


# In[13]:


wav_fname = os.path.join(DIRS['SONGS'], [x for x in os.listdir(DIRS['SONGS']) if x.endswith('.wav')][0])
wav_fname_new = wav_fname.replace('.wav', '_after.wav')


# low raw audio

# In[14]:


audio, _ = librosa.load(wav_fname, sr=M_PARAMS['SAMPLE_RATE'], mono=True)
audio[1000:1050]


# encode it to 8 bit amplitude

# In[15]:


quantized = mu_law_encode(audio, M_PARAMS['QUANTISATION_CHANNELS'])
quantized[1000:1050].eval(session=sess)


# get RNN input

# In[16]:


quantized_oh = _one_hot(quantized)
quantized_oh[0][1000:1020].eval(session=sess)


# let RNN out be exact RNN input (for test)
# 
# turn it back to 8 bit signal

# In[17]:


quantized_deoh = _de_one_hot(quantized_oh)
quantized_deoh[0][1000:1050].eval(session=sess)


# from 8 bit signal to real sound

# In[18]:


out = mu_law_decode(quantized_deoh,
    quantization_channels=M_PARAMS['QUANTISATION_CHANNELS'])
out[0][1000:1050].eval(session=sess)


# evaluate real_sound from tf to numpy

# In[19]:


out_wave = sess.run(out[0])


# write into file

# In[20]:


write_wav(out_wave, M_PARAMS['SAMPLE_RATE'], wav_fname_new)

