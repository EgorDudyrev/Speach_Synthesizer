def get_dirs(d=None):
    dirs = {'NOTEBOOKS': '/opt/notebooks/notebooks/',
            'SONGS': '/opt/notebooks/data/songs/'}
    return dirs[d] if d else dirs

def get_model_params(p=None):
    params = {'SAMPLE_RATE': 16000,
             'BATCH_SIZE': 1,
             'QUANTISATION_CHANNELS': 8}
    return params[p] if p else params