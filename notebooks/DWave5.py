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
        self.n_sparsified = 0
        
        self.define_constants()
        self.define_variables()
        self.define_placeholders()
        self.train_pass()
        self.define_loss()

    def define_constants(self):
        # Mask for masking W matrixes to c_t from input not connect to c_t output
        #M = np.ones(shape=(self.hidden_input, self.hidden_size), dtype='float32')
        #M[2,:self.hidden_size//2]=0
        #self.M = tf.constant(shape=(self.hidden_input, self.hidden_size), value=M)

        M = np.ones(shape=(self.hidden_size,self.hidden_input), dtype='float32')
        M[:self.hidden_size//2,2]=0
        self.M = tf.constant(shape=M.shape, value=M)
        
    def define_variables(self, dtype='float32'):
        def truncated_normal_init(shape, mean=0, stddev=0.01, dtype=dtype):
            return tf.truncated_normal(dtype=dtype, shape=shape, mean=mean, stddev=stddev)
        
        b0, b1 = self.block_shape

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        #shape = (self.hidden_input,self.hidden_size)
        #self.M_Ir, self.M_Iu, self.M_Ie = [tf.Variable(np.ones(shape=(shape[0]//b0,shape[1]//b1)), trainable=False, name=name) for name in ['M_Ir','M_Iu','M_Ie']]
        #self.IrV, self.IuV, self.IeV = [tf.Variable(truncated_normal_init(shape=(shape[0]*shape[1],)),name=name) for name in ['IrVS0','IuVS0','IeVS0']]
        #self.Ir, self.Iu, self.Ie = [self.get_sparse_tensor(M,V) for M,V in zip([self.M_Ir, self.M_Iu, self.M_Ie],[self.IrV, self.IuV, self.IeV])]

        shape = (self.hidden_size,self.hidden_input)
        self.M_Ir, self.M_Iu, self.M_Ie = [tf.Variable(np.ones(shape=(shape[0]//b0,shape[1]//b1), dtype='int64'), trainable=False, name=name) for name in ['M_Ir','M_Iu','M_Ie']]
        self.IrV, self.IuV, self.IeV = [tf.Variable(truncated_normal_init(shape=(shape[0]*shape[1],)),name=name) for name in ['IrV','IuV','IeV']]
        self.Ir, self.Iu, self.Ie = [self.get_sparse_tensor(M,V, is_input_layer=True) for M,V in zip([self.M_Ir, self.M_Iu, self.M_Ie],[self.IrV, self.IuV, self.IeV])]
        self.Ir, self.Iu, self.Ie = [tf.contrib.layers.dense_to_sparse(tf.multiply(tf.sparse.to_dense(t),self.M)) for t in [self.Ir, self.Iu, self.Ie]]

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        shape = (self.hidden_size,self.hidden_size)
        self.M_Rr, self.M_Ru, self.M_Re = [tf.Variable(np.ones(shape=(shape[0]//b0,shape[1]//b1), dtype='int64'), trainable=False, name=name) for name in ['M_Rr','M_Ru','M_Re']]
        self.RrV, self.RuV, self.ReV = [tf.Variable(truncated_normal_init(shape=(shape[0]*shape[1],)),name=name) for name in ['RrV','RuV','ReV']]
        self.Rr, self.Ru, self.Re = [self.get_sparse_tensor(M,V) for M,V in zip([self.M_Rr, self.M_Ru, self.M_Re],[self.RrV, self.RuV, self.ReV])]


        # Biases for hidden vectors of shape (hidden_size,)
        shape = (self.hidden_size,)
        self.br, self.bu, self.be = [tf.Variable(truncated_normal_init(shape=shape), name=name) for name in ['br','bu','be'] ]
        
        # O's matrices
        shape = (self.hidden_size//2,self.hidden_size//2)
        self.M_O1, self.M_O3 = [tf.Variable(np.ones(shape=(shape[0]//b0,shape[1]//b1), dtype='int64'), trainable=False, name=name) for name in ['M_O1','M_O3']]
        self.O1V, self.O3V = [tf.Variable(truncated_normal_init(shape=(shape[0]*shape[1],)),name=name) for name in ['O1V','O3V']]
        self.O1, self.O3 = [self.get_sparse_tensor(M,V) for M,V in zip([self.M_O1, self.M_O3],[self.O1V, self.O3V])]
        shape = (self.hidden_size//2,)
        self.bO1, self.bO3 = [tf.Variable(truncated_normal_init(shape=shape), name=name) for name in ['bO1','bO3'] ]

        #shape = (self.hidden_size//2,self.scale*2)
        shape = (self.scale*2,self.hidden_size//2)
        self.M_O2, self.M_O4 = [tf.Variable(np.ones(shape=(shape[0]//b0,shape[1]//b1), dtype='int64'), trainable=False, name=name) for name in ['M_O2','M_O4']]
        self.O2V, self.O4V = [tf.Variable(truncated_normal_init(shape=(shape[0]*shape[1],)),name=name) for name in ['O2V','O4V']]
        self.O2, self.O4 = [self.get_sparse_tensor(M,V) for M,V in zip([self.M_O2, self.M_O4],[self.O2V, self.O4V])]
        shape = (self.scale*2,)
        self.bO2, self.bO4 = [tf.Variable(truncated_normal_init(shape=shape), name=name) for name in ['bO2','bO4'] ]
    
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
        P_ct, P_ft = tf.expand_dims(self.P_ct_fl_unscaled, 1), tf.expand_dims(self.P_ft_fl_unscaled, 1)
        self.output_probs = tf.concat([P_ct, P_ft],2, name='output_probs')
        self.output_probs_fl = tf.reshape(self.output_probs, (-1,self.out_layer_size//2))
        
        self.output_true = tf.to_int32(self.Y_true*self.scale+self.scale)
        self.output_true_fl = tf.reshape(self.output_true, (-1,))
        
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_true_fl, logits=self.output_probs_fl, name='loss')
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        
    def train_pass(self):
        self.IuD, self.IrD, self.IeD, self.RuD, self.RrD, self.ReD = [tf.sparse.to_dense(t) for t in [self.Iu, self.Ir, self.Ie, self.Ru, self.Rr, self.Re]]


        self.h_t_trans = tf.scan(self.calc_hid, self.sound_trans, initializer=self.h_0, name='h_t_transposed')
        self.h_t = tf.transpose(self.h_t_trans, [1,0,2], name='h_t')
        self.h_c, self.h_f = tf.split(self.h_t_trans, num_or_size_splits=2, axis=2, name='split_h')
        

        self.h_c_flat, self.h_f_flat = tf.reshape(self.h_c, (-1,self.hidden_size//2)), tf.reshape(self.h_f, (-1, self.hidden_size//2))
        T = tf.transpose
        def sm(a,b):
            return tf.sparse_matmul(a,b, transpose_b=True, b_is_sparse=True)


        self.P_ct_fl_unscaled =  T(tf.sparse.matmul(self.O2, T(tf.nn.relu(T(tf.sparse.matmul(self.O1, T(self.h_c_flat)))+self.bO1))))+self.bO2
        self.P_ft_fl_unscaled =  T(tf.sparse.matmul(self.O4, T(tf.nn.relu(T(tf.sparse.matmul(self.O3, T(self.h_f_flat)))+self.bO3))))+self.bO4

        self.P_ct_fl, self.P_ft_fl = tf.nn.softmax(self.P_ct_fl_unscaled, 2), tf.nn.softmax(self.P_ft_fl_unscaled, 2)
        self.P_ct, self.P_ft = tf.reshape(self.P_ct_fl, (self.n_batches, -1, self.scale*2)), tf.reshape(self.P_ft_fl, (self.n_batches, -1, self.scale*2))        
        self.c_t, self.f_t = (tf.arg_max(self.P_ct,2)-self.scale)/self.scale, (tf.arg_max(self.P_ft,2)-self.scale)/self.scale
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

        T = tf.transpose
        
        h_tm1T = T(h_tm1)
        hid_inpT = T(self.hid_input)
        def sm(a,b):
            return tf.sparse_matmul(a,b, transpose_b=True, b_is_sparse=True)


        #u_t = tf.sigmoid(T(tf.sparse.matmul(self.Ru,h_tm1T) + tf.sparse.matmul(self.Iu,hid_inpT)) + self.bu, name='u_t')
        u_t = tf.sigmoid(sm(h_tm1, self.RuD) +  sm(self.hid_input, self.IuD)+ self.bu, name='u_t')
        #u_t = tf.sigmoid(T(tf.sparse.to_dense( tf.sparse.matmul(self.Ru,h_tm1T) + tf.sparse.matmul(self.Iu_masked,hid_inpT))) + self.bu, name='u_t')
        r_t = tf.sigmoid(sm(h_tm1, self.RrD) +  sm(self.hid_input, self.IrD)+ self.br, name='r_t')
        #r_t = tf.sigmoid(T(tf.sparse.matmul(self.Rr,h_tm1T) + tf.sparse.matmul(self.Ir,hid_inpT)) + self.br, name='r_t')
        # Definition of h~_t
        e_t = tf.tanh(tf.multiply(r_t, sm(h_tm1, self.ReD)) +sm(self.hid_input, self.IeD) + self.be, name='e_t')
        #e_t = tf.tanh(tf.multiply(r_t, T(tf.sparse.matmul(self.Re, h_tm1T))) +T(tf.sparse.matmul(self.Ie, hid_inpT))+ self.be, name='e_t')
        # Compute the next hidden state
        h_t = tf.multiply(u_t, h_tm1) + tf.multiply(1 - u_t, e_t)

        return h_t

    def sparsify(self, k, session, ):
        b0, b1 = self.block_shape
        ns = self.n_sparsified

        # Weights for input vectors of shape (input_dimensions, hidden_size)
        shape = (self.hidden_size,self.hidden_input)
        rMV, uMV, eMV = [self.get_sparse_matrix(k, t, shape) for t in [self.Ir, self.Iu, self.Ie]]
        self.M_Ir, self.M_Iu, self.M_Ie = [tf.assign(m,v[0]) for m,v in zip([self.M_Ir, self.M_Iu, self.M_Ie],[rMV, uMV, eMV])]
        vals = [session.run(v[1]) for v in [rMV, uMV, eMV]]
        #self.IrV, self.IuV, self.IeV = [tf.assign(m,v, validate_shape=False) for m,v in zip([self.IrV, self.IuV, self.IeV], vals)]
        self.IrV, self.IuV, self.IeV = [tf.Variable(v, name=name) for name,v in zip(['IrV', 'IuV', 'IeV'], vals)]
        session.run([t.initializer for t in [self.IrV, self.IuV, self.IeV]])
        self.Ir, self.Iu, self.Ie = [self.get_sparse_tensor(M,V, is_input_layer=True) for M,V in zip([self.M_Ir, self.M_Iu, self.M_Ie],[self.IrV, self.IuV, self.IeV])]
        self.Ir, self.Iu, self.Ie = [tf.contrib.layers.dense_to_sparse(tf.multiply(tf.sparse.to_dense(t),self.M)) for t in [self.Ir, self.Iu, self.Ie]]

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        shape = (self.hidden_size,self.hidden_size)
        rMV, uMV, eMV = [self.get_sparse_matrix(k, t, shape) for t in [self.Rr, self.Ru, self.Re]]
        self.M_Rr, self.M_Ru, self.M_Re = [tf.assign(m,v[0]) for m,v in zip([self.M_Rr, self.M_Ru, self.M_Re],[rMV, uMV, eMV])]
        vals = [session.run(v[1]) for v in [rMV, uMV, eMV]]
        #self.RrV, self.RuV, self.ReV = [tf.assign(m,v, validate_shape=False) for m,v in zip([self.RrV, self.RuV, self.ReV], vals)]
        self.RrV, self.RuV, self.ReV = [tf.Variable(v, name=name) for name,v in zip(['RrV','RuV', 'ReV'], vals)]
        session.run([t.initializer for t in [self.RrV, self.RuV, self.ReV]])
        self.Rr, self.Ru, self.Re = [self.get_sparse_tensor(M,V) for M,V in zip([self.M_Rr, self.M_Ru, self.M_Re],[self.RrV, self.RuV, self.ReV])]
        
        # O's matrices
        shape = (self.hidden_size//2,self.hidden_size//2)
        O1MV, O3MV = [self.get_sparse_matrix(k, t, shape) for t in [self.O1, self.O3]]
        self.M_O1, self.M_O3 = [tf.assign(m,v[0]) for m,v in zip([self.M_O1, self.M_O3],[O1MV, O3MV])]
        vals = [session.run(v[1]) for v in [O1MV, O3MV]]
        #self.O1V, self.O3V = [tf.assign(m,v, validate_shape=False) for m,v in zip([self.O1V, self.O3V], vals)]
        self.O1V, self.O3V = [tf.Variable(v, name=name) for name,v in zip(['O1V','O3V'], vals)]
        session.run([t.initializer for t in [self.O1V, self.O3V]])
        self.O1, self.O3 = [self.get_sparse_tensor(M,V) for M,V in zip([self.M_O1, self.M_O3],[self.O1V, self.O3V])]

        shape = (self.scale*2,self.hidden_size//2)
        O2MV, O4MV = [self.get_sparse_matrix(k, t, shape) for t in [self.O2, self.O4]]
        self.M_O2, self.M_O4 = [tf.assign(m,v[0]) for m,v in zip([self.M_O2,self.M_O4],[O2MV, O4MV])]
        vals = [session.run(v[1]) for v in [O2MV, O4MV]]
        #self.O2V, self.O4V = [tf.assign(m,v, validate_shape=False) for m,v in zip([self.O2V, self.O4V], vals)]
        self.O2V, self.O4V = [tf.Variable(v, name=name) for name,v in zip(['O2V','O4V'], vals)]
        session.run([t.initializer for t in [self.O2V, self.O4V]])
        self.O2, self.O4 = [self.get_sparse_tensor(M,V) for M,V in zip([self.M_O2, self.M_O4],[self.O2V, self.O4V])]

        self.n_sparsified +=1

        # updating graph
        self.train_pass()
        self.define_loss()

    
    def get_sparse_matrix(self, k, sparse_tensor, dense_matrix_shape):
        b0,b1 = self.block_shape
        n0, n1 = dense_matrix_shape
        spvd = tf.sparse.to_dense(sparse_tensor)
        shorted = tf.concat([[tf.split(t, spvd.shape[1]//b1, axis=1)] for t in tf.split(spvd, spvd.shape[0]//b0, axis=0)],0)
        shorted = tf.reshape(shorted, (-1,b0,b1))
        means = tf.reduce_mean(tf.abs(tf.reshape(shorted,(-1,b0*b1))),1)
        
        tops = tf.math.top_k(means, k=k)
        ids_flat = tops.indices
        ids = tf.transpose(tf.stack([ids_flat//(n1//b1), ids_flat%(n1//b1)]))
        M1 = tf.sparse.to_dense(tf.SparseTensor(indices=tf.cast(ids,'int64'),
                    values=np.ones(shape=ids.shape[0]).astype('int64'),
                    dense_shape=(n0//b0, n1//b1)), validate_indices=False)
        M1 = tf.cast(M1, 'int64')
        
        ids = tf.transpose(tf.stack([ids_flat//(n1//b1), ids_flat%(n1//b1)]))
        ids_flat_expand_start = ids[:,0]*n1*b0+ids[:,1]*b1
        ids_flat_expand = tf.reshape(tf.concat([
            tf.concat([ids_flat_expand_start+i*n1 for i in range(b0)],0)
                +j for j in range(b1)],0), (-1,))
        ids_flat_expand = tf.contrib.framework.sort(ids_flat_expand)
        vals = tf.gather(tf.reshape(spvd, (-1,)), ids_flat_expand)
        return M1, vals

    def get_sparse_tensor(self, sparse_matrix, var, is_input_layer=False):
        b0, b1 = self.block_shape
        k = tf.cast(var.shape[0]//b0//b1,'int32')
        n0 = sparse_matrix.shape[0]*b0
        n1 = sparse_matrix.shape[1]*b1
        
        tops = tf.math.top_k(tf.reshape(sparse_matrix,(-1,)), k=k)
        ids_flat = tf.cast(tops.indices, 'int64')
        ids = tf.transpose(tf.stack([ids_flat//(sparse_matrix.shape[1]), ids_flat%(sparse_matrix.shape[1])]))
        ids_flat_expand_start = ids[:,0]*n1*b0+ids[:,1]*b1
        ids_flat_expand = tf.reshape(tf.concat([
            tf.concat([ids_flat_expand_start+j for j in range(b1)],0)
                +i*n1 for i in range(b0)],0), (-1,))
        ids_flat_expand = tf.contrib.framework.sort(ids_flat_expand)
        #if is_input_layer:
        #    bl1 = tf.cast(ids_flat_expand<(2*n1), 'int32')
        #    bl2 = tf.cast(ids_flat_expand<(2*n1+n1//2), 'int32')
        #    print(tf.arg_max(bl1,0))
        #    print(tf.arg_max(bl2,0))
            #ids_flat_expand = tf.gather(ids_flat_expand, b1&b2)
        #    ids_flat_expand = ids_flat_expand[(bl1)&(bl2)]
            #ids_flat_expand = ids_flat_expand[]
        
        ids_expand = tf.transpose(tf.stack([ids_flat_expand//n1, ids_flat_expand%n1]))
        
        spv = tf.SparseTensor(indices=ids_expand, values=var, dense_shape=(n0,n1))
        return spv
            
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
        feed_dict={self.sound_X:sound_X, self.txt_embed_plh:txt_embed, self.Y_true:sound_Y}
        # Compute the losses
        _, train_loss = session.run([self.train_step, self.loss],
                                 feed_dict=feed_dict)
        return train_loss
    
    def generate(self, txt_emb, session, seconds=5, show_tqdm=False):
        if seconds is not None:
            pass
        
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
        gener_flat = gener_flat*128+128
        gener_flat = gener_flat[:,:,0]*256+gener_flat[:,:,1]
        gener_flat = gener_flat.T.flatten()