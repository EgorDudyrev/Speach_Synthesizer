import os
import tensorflow as tf
import librosa
from wavenet import AudioReader, mu_law_encode, mu_law_decode
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd
import re
import scipy.signal as sgn

def get_dirs(d=None, parent_dir='/opt/notebooks'):
    dirs = {
        'NOTEBOOKS': parent_dir+'/notebooks/',
        'SONGS': parent_dir+'/data/songs/',
        'RAW_DATA': parent_dir+'/raw_data/',
        'OUTPUT': parent_dir+'/output/',
        'MODELS': parent_dir+'/models/',
        'MODEL_CKPTS': parent_dir+'checkpoints/'
    }
    return dirs[d] if d else dirs

def get_model_params(p=None):
    params = {'SAMPLE_RATE': 16000,
             'BATCH_SIZE': 1,
             'QUANTIZATION_CHANNELS': 256*256}
    return params[p] if p else params

def _one_hot(input_batch, 
             batch_size=get_model_params('BATCH_SIZE'),
             quantization_channels=get_model_params('QUANTIZATION_CHANNELS')):
    '''One-hot encodes the waveform amplitudes.

    This allows the definition of the network as a categorical distribution
    over a finite set of possible amplitudes.
    '''
    with tf.name_scope('one_hot_encode'):
        encoded = tf.one_hot(
            input_batch,
            depth=quantization_channels,
            dtype=tf.float32)
        shape = [batch_size, -1, quantization_channels]
        encoded = tf.reshape(encoded, shape)
    return encoded

def _de_one_hot(encoded):
    '''One-hot decodes the waveform amplitudes.
    '''
    with tf.name_scope('one_hot_decode'):
        decoded = tf.argmax(encoded, axis=2)
    return decoded

def load_wav(filename, sample_rate=get_model_params('SAMPLE_RATE')):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    return audio
   
def write_wav(waveform, filename, sample_rate=get_model_params('SAMPLE_RATE'), verbose=False):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    if verbose: print('Updated wav file at {}'.format(filename))
    
def load_audio_not_one_hot(filename,
                      sample_rate=get_model_params('SAMPLE_RATE'),
                      quantization_channels=get_model_params('QUANTIZATION_CHANNELS'),
                      batch_size=get_model_params('BATCH_SIZE')
                     ):
    audio = load_wav(filename, sample_rate)
    quantized = mu_law_encode(audio, quantization_channels)
    return quantized

def load_audio_one_hot(filename,
                      sample_rate=get_model_params('SAMPLE_RATE'),
                      quantization_channels=get_model_params('QUANTIZATION_CHANNELS'),
                      batch_size=get_model_params('BATCH_SIZE')
                     ):
    quantized = load_audio_not_one_hot(filename, sample_rate, quantization_channels, batch_size)
    quantized_oh = _one_hot(quantized, batch_size, quantization_channels)
    return quantized_oh

def write_audio_not_one_hot(filename,
                        audio,
                        session,
                        sample_rate=get_model_params('SAMPLE_RATE'),
                        quantization_channels=get_model_params('QUANTIZATION_CHANNELS'),
                        verbose=False
                       ):
    out = mu_law_decode(audio, quantization_channels)
    out_wave = session.run(out)
    write_wav(out_wave, os.path.join(get_dirs('OUTPUT'), filename), sample_rate, verbose)
                           
def write_audio_one_hot(filename,
                        audio,
                        session,
                        sample_rate=get_model_params('SAMPLE_RATE'),
                        quantization_channels=get_model_params('QUANTIZATION_CHANNELS'),
                        verbose=False
                       ):
    quantized_deoh = _de_one_hot(audio)
    write_audio_not_one_hot(filname, quantized_deoh, session, sample_rate, quantization_channels, verbose)
    #out = mu_law_decode(quantized_deoh, quantization_channels)
    #out_wave = session.run(out[0])
    #write_wav(out_wave, os.path.join(get_dirs('OUTPUT'), filename), sample_rate, verbose)
    
def plot_losses(train_losses=None, validation_losses=None, title=None):
    if train_losses is None and validation_losses is None:
        return
    if train_losses is not None:
        plt.plot(train_losses, '-b', label='Train loss')
    if validation_losses is not None:
        plt.plot(validation_losses, '-r', label='Validation loss')
    plt.legend(loc=0)
    title = 'Loss' if title is None else title
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
def get_train_test(X, batch_size, truncated_len, session=None, text_oh=None):
    X_train, Y_train = generate_batch(X, batch_size, truncated_len, text_oh=text_oh)
    X_test, Y_test = generate_batch(X, batch_size, truncated_len, text_oh=text_oh)
    if session is not None:
        X_train, Y_train, X_test, Y_test = session.run([X_train, Y_train, X_test, Y_test])
    return X_train, Y_train, X_test, Y_test

def generate_batch(X, batch_size, truncated_len, next_c_unknown=True, text_oh=None):
    idxs = random.randint(0,X.shape[1]-truncated_len-2, size=batch_size)
    x = tf.concat([X[:,idxs[i]:idxs[i]+truncated_len+1] for i in range(batch_size)],axis=0)
    if next_c_unknown:
        x = tf.concat([x, 
           np.zeros((batch_size,truncated_len+1,1)),
          ],axis=2)[:,:-1:,:]
    else:
        x = tf.concat([x, 
                tf.concat([tf.reshape(x[:,1:,0],(batch_size,-1,1)), [[[0]]]*batch_size ],axis=1)], axis=2
            )[:,:-1,:]
    if text_oh is not None:
        toh = tf.concat([text_oh[:,idxs[i]:idxs[i]+truncated_len] for i in range(batch_size)],axis=0)
        x = tf.concat([x, toh],axis=2)
    y = tf.concat([X[:,idxs[i]+1:idxs[i]+truncated_len+1] for i in range(batch_size)],axis=0)
    return x,y

def load_data(files, n_files=None, quantization_channels=65536):
    Xs = []
    for idx, fname in enumerate(files):
        if n_files is not None and idx==n_files: break
        if type(fname)==tuple:
            fname = fname[1]
        if type(fname)!=str:
            fname = fname.as_posix()
        audio = load_audio_not_one_hot(fname, quantization_channels=quantization_channels)
        Xs.append(audio[:-1])
    X = tf.concat(Xs, axis=0)
    X = tf.reshape(X, (1,-1,1))
    
    nbits = int(np.log2(quantization_channels))
    n = 2**(nbits//2)
    X = tf.concat([X//n, X%n], 2)
    X = (X-n//2)/(n//2)
    return tf.identity(X, name="X_data")

def split_on_vowels(s, vowels='аеёиоуыэюя'):
    vowel_idxs = np.array([(m.start(0), m.end(0)) for m in re.finditer('|'.join(vowels), s)])
    if len(vowel_idxs)<=1:
        return [s]
    b = vowel_idxs[:-1,1]!=vowel_idxs[1:,0]
    vowel_idxs[:-1][b,1] = (vowel_idxs[:-1][b,1]+vowel_idxs[1:][b,0])/2
    vowel_idxs[1:][b,0] = vowel_idxs[:-1][b,1]
    vowel_idxs[0,0] = 0
    vowel_idxs[-1,1] = len(s)
    return [s[idx[0]:idx[1]] for idx in vowel_idxs]

def force_align(audio_roll, txt_vow, lim):
    itr = 0
    left,right = 1, len(audio_roll)
    d = len(audio_roll)//len(txt_vow)
    d1 = 0
    idxs = None
    while itr < 50 and abs(d-d1)>0:
        idxs = sgn.find_peaks(audio_roll, distance=d, height=lim)[0]
        d1 = d
        if len(idxs) == len(txt_vow):
            break
        if len(idxs) > len(txt_vow):
            d = (d+right)//2
            left = d1
        if len(idxs) < len(txt_vow):
            d = (left+d)//2
            right = d1
        itr +=1
    else:
        return None
        #raise ValueError(f'Cannot force align file {fname}')
    return idxs

def plot_audio(audio=None, audio_roll=None, lim=None, start_stops=None, txt=None, grid=False):
    if audio is not None:
        plt.plot(audio, alpha=0.3 if audio_roll is not None else 1, color='lightblue')
    if audio_roll is not None:
        plt.plot(audio_roll, color='orange')
    if lim is not None:
        plt.axhline(lim, alpha=1)
    if start_stops is not None:
        for s in start_stops[:,1].astype(int):
            plt.axvline(s, color='green', linewidth=1)
        for s in start_stops[:,2].astype(int):
            plt.axvline(s, color='red', linewidth=1)
    if start_stops is not None and txt is not None:
        plt.xticks(start_stops[:,1:3].astype(int).mean(1), txt if type(txt)==list else txt.split(' '), rotation=45)
    plt.yticks(np.linspace(audio.min(), audio.max(), 10),np.linspace(audio.min(), audio.max(), 10).round(1))
    if grid: plt.grid()

def align_audio(audio, txt, window_size=1000):
    audio_ev_norm = (audio-np.median(audio))/audio.std()
    audio_ev_norm = abs(audio_ev_norm)

    audio_roll = pd.Series(audio_ev_norm).rolling(window=window_size).mean().iloc[window_size-1:].values
    audio_roll = np.resize(audio_roll, (len(audio)))
    lim = np.median(audio_roll)

    txt_vow = np.concatenate([split_on_vowels(x)  for x in txt.split(' ')])

    idxs = force_align(audio_roll, txt_vow, lim)
    if idxs is None:
        return audio_ev_norm, audio_roll, lim, None

    _,_,word_start, word_stop = sgn.peak_widths(audio_roll, idxs)
    start_stops = np.hstack([word_start.reshape((-1,1)),word_stop.reshape((-1,1))])
    start_stops = np.concatenate([sorted(start_stops, key=lambda x: x[0])])

    b = start_stops[:-1,1]>start_stops[1:,0]
    start_stops[:-1][b,1] = start_stops[1:][b,0]
    b = (start_stops[1:,1]-start_stops[1:,0])<500
    start_stops[1:][b,0] = start_stops[:-1][b,1]

    word_idxs = np.concatenate([[idx]*len(split_on_vowels(t)) for idx,t in enumerate(txt.split(' '))])
    word_idxs is None


    m = np.hstack([word_idxs.reshape(-1,1), txt_vow.reshape(-1,1), start_stops])
    m = sorted([(''.join(m[m[:,0]==x][:,1]), int(float(m[m[:,0]==x][0,2])),int(float(m[m[:,0]==x][-1,3]))) for x in sorted(set(m[:,0]))], key=lambda x: x[1])
    m = np.array(m)
    
    m_spaced = m.copy()
    m_spaced = np.concatenate([[[' ','0', m_spaced[0,1]]],
                               m_spaced,
                               [[' ', m_spaced[-1,2], max(int(m_spaced[-1,2]),audio_roll.shape[0])]]])
    t = np.hstack([m_spaced[:-1,2].reshape(-1,1), m_spaced[1:,1].reshape(-1,1)])
    if sum(t[:,1]>t[:,0])>0:
        t = t[t[:,1]>t[:,0]]
        m_spaced = np.concatenate([m_spaced, [[' ',left, right] for left,right in t]])
    m_spaced = np.concatenate([sorted(m_spaced, key=lambda x: int(x[1]))])
    
    return audio_ev_norm, audio_roll, lim, m_spaced

def load_text_oh(files, n_files=None, max_word_len=35):
    char_to_int = {k: idx for idx,k in enumerate('\0 абвгдежзийклмнопрстуфхцчшщъыьэюяё')}
    tas = []
    for idx, fname in enumerate(files):
        if n_files is not None and idx==n_files: break
        if type(fname)==tuple:
            fname = fname[1]
        if type(fname)!=str:
            fname = fname.as_posix()
        with open(fname, 'rb') as f:
            ta = np.load(f)
        tas.append(ta)
    ta = np.concatenate(tas)
    words = ta[:,0]
    words = [[char_to_int[c] for c in x]+[0]*(max_word_len-len(x)) for x in words]
    counts = ta[:,2].astype(int)-ta[:,1].astype(int)
    shaped_text = np.concatenate([([w]*counts[idx]) for idx, w in enumerate(words) if counts[idx]>0])
    
    oh = tf.one_hot(shaped_text, len(char_to_int), dtype='float64')
    oh = tf.reshape(oh, (1,-1,len(char_to_int)*max_word_len))
    return oh