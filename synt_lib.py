import os
import tensorflow as tf
import librosa
from wavenet import AudioReader, mu_law_encode, mu_law_decode
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

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
    
def get_train_test(X, batch_size, truncated_len, session=None):
    X_train, Y_train = generate_batch(X, batch_size, truncated_len)
    X_test, Y_test = generate_batch(X, batch_size, truncated_len)
    if session is not None:
        X_train, Y_train, X_test, Y_test = session.run([X_train, Y_train, X_test, Y_test])
    return X_train, Y_train, X_test, Y_test

def generate_batch(X, batch_size, truncated_len, next_c_unknown=True):
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
    y = tf.concat([X[:,idxs[i]+1:idxs[i]+truncated_len+1] for i in range(batch_size)],axis=0)
    return x,y

def load_data(files, n_files=None, quantization_channels=65536):
    Xs = []
    for idx, fname in enumerate(files):
        if n_files is not None and idx==n_files: break
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
