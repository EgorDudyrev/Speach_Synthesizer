import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook

class WaveGRU():
    def __init__(self, input_dim=3, hidden_input=128, hidden_size=224, out_layer_size=16,
                 block_shape=(16,1), vocab_size=35, max_text_len=100, sample_rate=16000,
                n_batches=5):
        self.input_dim = input_dim
        self.hidden_input = hidden_input
        self.text_embed_size = hidden_input-input_dim
        self.hidden_size = hidden_size
        self.out_layer_size=out_layer_size
        self.scale = (2**(self.out_layer_size//2))//2
        self.block_shape = block_shape
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.sample_rate = sample_rate
        self.n_batches = n_batches
        
        self.define_constants()
        self.define_variables()
        self.define_placeholders()
        self.train_pass()
        self.define_loss()
        
    def define_constants(self):
        # Mask for masking W matrixes to c_t from input not connect to c_t output
        M = np.ones(shape=(self.hidden_input, self.hidden_size), dtype='float32')
        M[2,:self.hidden_size//2]=0
        self.M = tf.constant(shape=(self.hidden_input, self.hidden_size), value=M)
        
    def define_variables(self, dtype='float32'):     
        def truncated_normal_init(shape, mean=0, stddev=0.01, dtype=dtype):
            return tf.truncated_normal(dtype=dtype, shape=shape, mean=mean, stddev=stddev)

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        shape=(self.hidden_input, self.hidden_size)
        self.Ir, self.Iu, self.Ie = [tf.Variable(truncated_normal_init(shape), name=name) for name in ['Ir','Iu','Ie']]
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        shape=(self.hidden_size,self.hidden_size)
        self.Rr, self.Ru, self.Re = [tf.Variable(truncated_normal_init(shape), name=name) for name in ['Rr','Ru','Re']]
        # Biases for hidden vectors of shape (hidden_size,)
        shape=(self.hidden_size,)
        self.br, self.bu, self.be = [tf.Variable(truncated_normal_init(shape), name=name) for name in ['br','bu','be']]
        # O's matrices
        shape=(self.hidden_size//2, self.hidden_size//2)
        self.O1, self.O3 = [tf.Variable(truncated_normal_init(shape), name=name) for name in ['O1','O3',]]
        shape=(self.hidden_size//2, self.scale*2)
        self.O2, self.O4 = [tf.Variable(truncated_normal_init(shape), name=name) for name in ['O2', 'O4']]
        # biases for O's
        shape=(self.hidden_size//2,)
        self.bO1, self.bO3 = [tf.Variable(truncated_normal_init(shape), name=name) for name in ['bO1','bO3']]
        shape=(self.scale*2,)
        self.bO2, self.bO4 = [tf.Variable(truncated_normal_init(shape), name=name) for name in ['bO2','bO4']]
        
        #sparse matrices
        m1,m2 = self.block_shape
        self.M_Ir, self.M_Iu, self.M_Ie = [np.ones(shape=(self.hidden_input//m1, self.hidden_size//m2),dtype=dtype) for i in range(3)]
        self.M_Rr, self.M_Ru, self.M_Re = [np.ones(shape=(self.hidden_size//m1, self.hidden_size//m2),dtype=dtype) for i in range(3)]
        self.M_O1, self.M_O3 = [np.ones(shape=(self.hidden_size//2//m1, self.hidden_size//2//m2),dtype=dtype) for i in range(2)]
        self.M_O2, self.M_O4 = [np.ones(shape=(self.hidden_size//2//m1, self.scale*2//m2),dtype=dtype) for i in range(2)]
        
        self.M_IrV, self.M_IuV, self.M_IeV, self.M_RrV, self.M_RuV, self.M_ReV, \
        self.M_O1V, self.M_O2V, self.M_O3V, self.M_O4V = [
            tf.Variable(v, name=n, trainable=False) for v, n in zip([
                self.M_Ir, self.M_Iu, self.M_Ie, self.M_Rr, self.M_Ru, self.M_Re,
                self.M_O1, self.M_O2, self.M_O3, self.M_O4,],[
                'M_IrV','M_IuV','M_IeV', 'M_RrV','M_RuV','M_ReV',
                'M_O1V','M_O2V','M_O3V', 'M_O4V',])
        ]
    
    def restore_variables(self, variables): 
        self.Ir, self.Iu, self.Ie = [tf.assign(t,variables[name+':0']) for t,name in zip([self.Ir, self.Iu, self.Ie],['Ir','Iu','Ie'])]
        self.Rr, self.Ru, self.Re = [tf.assign(t,variables[name+':0']) for t,name in zip([self.Rr, self.Ru, self.Re],['Rr','Ru','Re'])]
        self.br, self.bu, self.be = [tf.assign(t,variables[name+':0']) for t,name in zip([self.br, self.bu, self.be],['br','bu','be'])]
        
        self.O1, self.O2, self.O3, self.O4 = [tf.assign(t,variables[name+':0']) for t,name in zip([self.O1, self.O2, self.O3, self.O4],['O1','O2','O3','O4'])]
        self.bO1, self.bO2, self.bO3, self.bO4 = [tf.assign(t,variables[name+':0']) for t,name in zip([self.bO1, self.bO2, self.bO3, self.bO4],['bO1','bO2','bO3','bO4'])]
        
        self.M_Ir, self.M_Iu, self.M_Ie = [variables[name+':0'] for name in ['M_IrV','M_IuV','M_IeV']]
        self.M_Rr, self.M_Ru, self.M_Re = [variables[name+':0'] for name in ['M_IrV','M_IuV','M_IeV']]
        self.M_O1, self.M_O2, self.M_O3, self.M_O4 = [variables[name+':0'] for name in ['M_O1V','M_O2V','M_O3V','M_O4V']]
                                        
        self.M_IrV, self.M_IuV, self.M_IeV, self.M_RrV, self.M_RuV, self.M_ReV, \
        self.M_O1V, self.M_O2V, self.M_O3V, self.M_O4V = [
            tf.assign(t, variables[name+':0']) for t, name in zip([
                self.M_Ir, self.M_Iu, self.M_Ie, self.M_Rr, self.M_Ru, self.M_Re,
                self.M_O1, self.M_O2, self.M_O3, self.M_O4,],[
                'M_IrV','M_IuV','M_IeV', 'M_RrV','M_RuV','M_ReV',
                'M_O1V','M_O2V','M_O3V', 'M_O4V',])
        ]
        
    def define_placeholders(self):
        self.sound_X = tf.placeholder(dtype=tf.float32, shape=(self.n_batches, None, self.input_dim), name='soundInput')
        self.sound_trans = tf.transpose(self.sound_X, [1,0,2], name='sound_trans')
        self.Y_true = tf.placeholder(dtype=tf.float32, shape=(self.n_batches,None, 2), name='Y_true')
        self.sound_tm1 = tf.placeholder(dtype=tf.float32, shape=(self.n_batches, 1,self.input_dim), name='sound_tm1')
        h_0 = tf.matmul(
            a=self.sound_trans[0, :, :], name='h_0',
            b=tf.zeros(dtype=tf.float32, shape=(self.input_dim, self.hidden_size)),
        )
        self.h_0 = tf.placeholder_with_default(h_0, shape=(self.n_batches,self.hidden_size),name='h_0_placeholder')
        self.txt_embed_plh = tf.placeholder(dtype=tf.float32, shape=(1,self.text_embed_size))
        self.txt_embed_tile = tf.tile(self.txt_embed_plh, [self.n_batches, 1])
    
    def define_loss(self):
        #transpose to [batch, time, coarse/fine]
        P_ct, P_ft = tf.expand_dims(self.P_ct_fl_unscaled, 1), tf.expand_dims(self.P_ft_fl_unscaled, 1)
        self.output_probs = tf.concat([P_ct, P_ft],2, name='output_probs')
        self.output_probs_fl = tf.reshape(self.output_probs, (-1,self.out_layer_size//2))
        
        self.output_true = tf.to_int32(self.Y_true*self.scale+self.scale)
        self.output_true_fl = tf.reshape(self.output_true, (-1,))
        
        
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_true_fl, logits=self.output_probs_fl, name='loss')
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        
    
    def train_pass(self):
        """M_Iu, M_Ir, M_Ie = [self.extend_matrix(M) for M in [self.M_Iu, self.M_Ir, self.M_Ie]]
        M_Ru, M_Rr, M_Re = [self.extend_matrix(M) for M in [self.M_Ru, self.M_Rr, self.M_Re]]
        
        self.Iu = tf.assign(self.Iu, tf.multiply(self.Iu, M_Iu, name='Iu_sparse'))
        self.Ir = tf.assign(self.Ir, tf.multiply(self.Ir, M_Ir, name='Ir_sparse'))
        self.Ie = tf.assign(self.Ie, tf.multiply(self.Ie, M_Ie, name='Ie_sparse'))
        """
        self.Iu_masked = tf.multiply(self.Iu, self.M, name='Iu_mask')
        self.Ir_masked = tf.multiply(self.Ir, self.M, name='Ir_mask')
        self.Ie_masked = tf.multiply(self.Ie, self.M, name='Ie_mask')
        """
        #self.Iu_sparse, self.Ir_sparse, self.Ie_sparse = self.Iu_masked, self.Ir_masked, self.Ie_masked
        self.Ru = tf.assign(self.Ru, tf.multiply(self.Ru, M_Ru, name='Ru_sparse'))
        self.Rr = tf.assign(self.Rr, tf.multiply(self.Rr, M_Rr, name='Rr_sparse'))
        self.Re = tf.assign(self.Re, tf.multiply(self.Re, M_Re, name='Re_sparse'))
        
        M_O1, M_O2 = [self.extend_matrix(M) for M in [self.M_O1, self.M_O2]]
        self.O1_sparse, self.O2_sparse = tf.multiply(self.O1, M_O1, name='O1_sparse'), tf.multiply(self.O2, M_O2, name='O2_sparse')
        self.O1, self.O2 = tf.assign(self.O1, self.O1_sparse), tf.assign(self.O2, self.O2_sparse)
        del M_O1, M_O2
        M_O3, M_O4 = [self.extend_matrix(M) for M in [self.M_O3, self.M_O4]]
        self.O3_sparse, self.O4_sparse = tf.multiply(self.O3, M_O3, name='O3_sparse'), tf.multiply(self.O4, M_O4, name='O4_sparse')
        self.O3, self.O4 = tf.assign(self.O3, self.O3_sparse), tf.assign(self.O4, self.O4_sparse)
        del M_O3, M_O4
        """

        
        self.h_t_trans = tf.scan(self.calc_hid, self.sound_trans, initializer=self.h_0, name='h_t_transposed')
        self.h_t = tf.transpose(self.h_t_trans, [1,0,2], name='h_t')
        self.h_c, self.h_f = tf.split(self.h_t_trans, num_or_size_splits=2, axis=2, name='split_h')
        
        self.h_c_flat, self.h_f_flat = tf.reshape(self.h_c, (-1,self.hidden_size//2)), tf.reshape(self.h_f, (-1, self.hidden_size//2))
        self.P_ct_fl_unscaled = tf.matmul(tf.nn.relu(tf.matmul(self.h_c_flat, self.O1)+self.bO1), self.O2)+self.bO2
        self.P_ft_fl_unscaled = tf.matmul(tf.nn.relu(tf.matmul(self.h_f_flat, self.O3)+self.bO3), self.O4)+self.bO4
        self.P_ct_fl, self.P_ft_fl = tf.nn.softmax(self.P_ct_fl_unscaled, 2), tf.nn.softmax(self.P_ft_fl_unscaled, 2)
        self.P_ct, self.P_ft = tf.reshape(self.P_ct_fl, (self.n_batches, -1, self.scale*2)), tf.reshape(self.P_ft_fl, (self.n_batches, -1, self.scale*2))        
        self.c_t, self.f_t = tf.arg_max(self.P_ct,2), tf.arg_max(self.P_ft,2)
        self.c_t, self.f_t = tf.expand_dims(self.c_t,2), tf.expand_dims(self.f_t,2)
        
        self.y = tf.concat([self.c_t, self.f_t], axis=2, name='y')
        
    def calc_hid(self, h_tm1, sound_tm1):
        """Calculate hidden state
        Arguments
        ---------
        h_tm1: np.matrix
            The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix
            The input vector.
        """
        self.hid_input = tf.concat([sound_tm1, self.txt_embed_tile],axis=1)
                 
    
        u_t = tf.sigmoid(tf.matmul(h_tm1, self.Ru) + tf.matmul(self.hid_input, self.Iu_masked) + self.bu, name='u_t')
        r_t = tf.sigmoid(tf.matmul(h_tm1, self.Rr) + tf.matmul(self.hid_input, self.Ir_masked) + self.br, name='r_t')
        # Definition of h~_t
        e_t = tf.tanh(tf.multiply(r_t, tf.matmul(h_tm1, self.Re))+tf.matmul(self.hid_input, self.Ie_masked)+ self.be, name='e_t')
        # Compute the next hidden state
        h_t = tf.multiply(u_t, h_tm1) + tf.multiply(1 - u_t, e_t)
        
        return h_t
    
    def extend_matrix(self, M):
        coefs = self.block_shape
        return np.concatenate([[np.concatenate([[j]*coefs[1] for j in i])]*coefs[0] for i in M])                
    
    def sparsify(self, k, session):
        self.M_Iu, self.M_Ir, self.M_Ie = [
            self.get_sparse_matrix(t, k, session) for t in [self.Iu, self.Ir, self.Ie]]
        self.M_Ru, self.M_Rr, self.M_Re = [
            self.get_sparse_matrix(t, k, session) for t in [self.Ru, self.Rr, self.Re]]
        self.M_O1, self.M_O2, self.M_O3, self.M_O4 = [
            self.get_sparse_matrix(t, k, session) for t in [self.O1, self.O2, self.O3, self.O4]]
        
        self.M_IrV, self.M_IuV, self.M_IeV, self.M_RrV, self.M_RuV, self.M_ReV, \
        self.M_O1V, self.M_O2V, self.M_O3V, self.M_O4V = [
            tf.assign(t,v) for t,v in zip([
                self.M_IrV, self.M_IuV, self.M_IeV, self.M_RrV, self.M_RuV, self.M_ReV,
                self.M_O1V, self.M_O2V, self.M_O3V, self.M_O4V
            ],[ self.M_Iu, self.M_Ir, self.M_Ie, self.M_Ru, self.M_Rr, self.M_Re,
                self.M_O1, self.M_O2, self.M_O3, self.M_O4,
            ])
        ]
        
        M_Iu, M_Ir, M_Ie = [self.extend_matrix(M) for M in [self.M_Iu, self.M_Ir, self.M_Ie]]
        M_Ru, M_Rr, M_Re = [self.extend_matrix(M) for M in [self.M_Ru, self.M_Rr, self.M_Re]]
        
        self.Iu = tf.assign(self.Iu, tf.multiply(self.Iu, M_Iu, name='Iu_sparse'))
        self.Ir = tf.assign(self.Ir, tf.multiply(self.Ir, M_Ir, name='Ir_sparse'))
        self.Ie = tf.assign(self.Ie, tf.multiply(self.Ie, M_Ie, name='Ie_sparse'))
        #self.Iu_sparse, self.Ir_sparse, self.Ie_sparse = self.Iu_masked, self.Ir_masked, self.Ie_masked
        self.Ru = tf.assign(self.Ru, tf.multiply(self.Ru, M_Ru, name='Ru_sparse'))
        self.Rr = tf.assign(self.Rr, tf.multiply(self.Rr, M_Rr, name='Rr_sparse'))
        self.Re = tf.assign(self.Re, tf.multiply(self.Re, M_Re, name='Re_sparse'))
        
        M_O1, M_O2 = [self.extend_matrix(M) for M in [self.M_O1, self.M_O2]]
        self.O1_sparse, self.O2_sparse = tf.multiply(self.O1, M_O1, name='O1_sparse'), tf.multiply(self.O2, M_O2, name='O2_sparse')
        self.O1, self.O2 = tf.assign(self.O1, self.O1_sparse), tf.assign(self.O2, self.O2_sparse)
        del M_O1, M_O2
        M_O3, M_O4 = [self.extend_matrix(M) for M in [self.M_O3, self.M_O4]]
        self.O3_sparse, self.O4_sparse = tf.multiply(self.O3, M_O3, name='O3_sparse'), tf.multiply(self.O4, M_O4, name='O4_sparse')
        self.O3, self.O4 = tf.assign(self.O3, self.O3_sparse), tf.assign(self.O4, self.O4_sparse)
        del M_O3, M_O4
            
    def get_sparse_matrix(self, tensor, k, session):
        tensor_ev = tensor.eval(session=session)
        m1,m2 = self.block_shape
        shorted = np.array([[
            tensor_ev[m1*i:m1*(i+1),m2*j:m2*(j+1)].mean()
            for j in range(tensor_ev.shape[1]//m2)] for i in range(tensor_ev.shape[0]//m1)]
        )
        flat = shorted.flatten()
        k_ = int(k*int(flat.shape[0]))
        idxs = flat.argsort()[:k_]
        M = np.ones((int(flat.shape[0])))
        M[idxs] = 0
        return M.reshape(shorted.shape)
            
    @staticmethod
    def calc_sparsity_level(t,sparsify_epochs ,sparsity_level):
        t0 = min(sparsify_epochs)
        S = max(sparsify_epochs)-t0
        Z = sparsity_level
        return Z*(1-(1-(t-t0)/S)**3)
    
    def train(self, sound_X, txt_embed, session):
        sound_Y = sound_X
        c = sound_X[:,:,0]
        c_shift = np.hstack([c, [[0]]*self.n_batches])[:,1:]
        c_shift = c_shift.reshape(self.n_batches,-1,1)
        sound_X = np.concatenate([sound_X, c_shift],2)
        
        #txt_embed = session.run(self.txt_embed, feed_dict={self.text_X:txt_X})
        feed_dict={gru.sound_X:sound_X, gru.txt_embed_plh:txt_embed, gru.Y_true:sound_Y}
        # Compute the losses
        _, train_loss = session.run([self.train_step, self.loss],
                                 feed_dict=feed_dict)
        return train_loss
    
    def generate(self, txt_emb, session, seconds=5, show_tqdm=False):
        if seconds is not None:
            pass
        

        M_Iu, M_Ir, M_Ie = [self.extend_matrix(M) for M in [self.M_Iu, self.M_Ir, self.M_Ie]]
        M_Ru, M_Rr, M_Re = [self.extend_matrix(M) for M in [self.M_Ru, self.M_Rr, self.M_Re]]
        self.Iu = tf.assign(self.Iu, tf.multiply(self.Iu, M_Iu, name='Iu_sparse'))
        self.Ir = tf.assign(self.Ir, tf.multiply(self.Ir, M_Ir, name='Ir_sparse'))
        self.Ie = tf.assign(self.Ie, tf.multiply(self.Ie, M_Ie, name='Ie_sparse'))
        #self.Iu_sparse, self.Ir_sparse, self.Ie_sparse = self.Iu_masked, self.Ir_masked, self.Ie_masked
        self.Ru = tf.assign(self.Ru, tf.multiply(self.Ru, M_Ru, name='Ru_sparse'))
        self.Rr = tf.assign(self.Rr, tf.multiply(self.Rr, M_Rr, name='Rr_sparse'))
        self.Re = tf.assign(self.Re, tf.multiply(self.Re, M_Re, name='Re_sparse'))

        M_O1, M_O2 = [self.extend_matrix(M) for M in [self.M_O1, self.M_O2]]
        self.O1_sparse, self.O2_sparse = tf.multiply(self.O1, M_O1, name='O1_sparse'), tf.multiply(self.O2, M_O2, name='O2_sparse')
        self.O1, self.O2 = tf.assign(self.O1, self.O1_sparse), tf.assign(self.O2, self.O2_sparse)
        del M_O1, M_O2
        M_O3, M_O4 = [self.extend_matrix(M) for M in [self.M_O3, self.M_O4]]
        self.O3_sparse, self.O4_sparse = tf.multiply(self.O3, M_O3, name='O3_sparse'), tf.multiply(self.O4, M_O4, name='O4_sparse')
        self.O3, self.O4 = tf.assign(self.O3, self.O3_sparse), tf.assign(self.O4, self.O4_sparse)
        del M_O3, M_O4
        
        h_t = np.zeros((self.n_batches, self.hidden_size), dtype='float32')
        x = np.zeros((self.n_batches, 1, self.input_dim), dtype='float32')
        
        xs = [x[:,:,:2]]
        n_iters = int(seconds*self.sample_rate/self.n_batches)
        for i in tqdm_notebook(range(n_iters), disable=not show_tqdm):
            feed_dict={self.sound_X:x, self.h_0:h_t, self.txt_embed_plh:txt_emb}
            c_t = session.run(self.c_t, feed_dict=feed_dict)
            x[:,0,2] = c_t[:,0,0]
            feed_dict={self.sound_X:x, self.h_0:h_t, self.txt_embed_plh:txt_emb}
            x, h_t = session.run([self.y, self.h_t_trans], feed_dict=feed_dict) 
            x = np.concatenate([x,np.zeros((self.n_batches,1,1))],2)
            h_t = h_t[0] #[1,n_batches,hidden_size] -> [n_batches, hidden_size]
            xs.append(x[:,:,:2])
        return xs
    
    def generate_audio(self, txt_emb, session, seconds=5, show_tqdm=False):
        gener = self.generate(txt_emb, session, seconds, show_tqdm=show_tqdm)
        gener_flat = np.concatenate(gener,1)
        gener_flat = gener_flat*self.scale+self.scale
        gener_flat = gener_flat[:,:,0]*2*self.scale+gener_flat[:,:,1]
        gener_flat = gener_flat.T.flatten()