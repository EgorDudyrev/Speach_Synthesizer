import os
import tensorflow as tf
import librosa
from wavenet import AudioReader, mu_law_encode, mu_law_decode
import numpy as np

def get_dirs(d=None):
    dirs = {'NOTEBOOKS': '/opt/notebooks/notebooks/',
            'SONGS': '/opt/notebooks/data/songs/',
            'RAW_DATA': '/opt/notebooks/raw_data/',
            'OUTPUT': '/opt/notebooks/output/'}
    return dirs[d] if d else dirs

def get_model_params(p=None):
    params = {'SAMPLE_RATE': 16000,
             'BATCH_SIZE': 1,
             'QUANTIZATION_CHANNELS': 8}
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