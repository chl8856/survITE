import tensorflow as tf
import numpy as np

from utils.ipm_utils import mmd2_lin, wasserstein


_EPSILON = 1e-08

################################
##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + _EPSILON)

def div(x, y):
    return tf.div(x, (y + _EPSILON))


##### NETWORK FUNCTIONS
def fcnet(x_, o_dim_, o_fn_, num_layers_=1, h_dim_=100, activation_fn=tf.nn.relu, keep_prob_=1.0, w_reg_=None, name='fcnet', reuse=tf.AUTO_REUSE):
    '''
        x_            : (2D-tensor) input
        o_dim_        : (int) output dimension
        o_type_       : (string) output type one of {'continuous', 'categorical', 'binary'}
        num_layers_   : (int) # of hidden layers
        activation_fn_: tf activation functions
        reuse         : (bool) 
    '''
    with tf.variable_scope(name, reuse=reuse):
        if num_layers_ == 1:
            out =  tf.contrib.layers.fully_connected(inputs=x_, num_outputs=o_dim_, activation_fn=o_fn_, weights_regularizer=w_reg_, scope='layer_out')
        else:
            for tmp_layer in range(num_layers_-1):
                if tmp_layer == 0:
                    net = x_
                net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=h_dim_, activation_fn=activation_fn, weights_regularizer=w_reg_, scope='layer_'+str(tmp_layer))
                net = tf.nn.dropout(net, keep_prob=keep_prob_)
            out =  tf.contrib.layers.fully_connected(inputs=net, num_outputs=o_dim_, activation_fn=o_fn_, weights_regularizer=w_reg_, scope='layer_out')  
    return out


################################
##### NETWORK 
class SurvITE:
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess               = sess
        self.name               = name

        ### INPUT DIMENSIONS
        self.x_dim              = input_dims['x_dim']
        self.t_max              = input_dims['t_max']
        self.num_Event          = input_dims['num_Event'] #Without counting censoring.
        

        ### NETWORK HYPER-PARMETERS
        self.z_dim              = network_settings['z_dim']  #PHI(X)
        
        self.h_dim1             = network_settings['h_dim1']  #PHI
        self.h_dim2             = network_settings['h_dim2']  #Hypothesis
        
        self.num_layers1        = network_settings['num_layers1']
        self.num_layers2        = network_settings['num_layers2']
 
        self.active_fn          = network_settings['active_fn']
        self.reg_scale          = network_settings['reg_scale']
        
        self.ipm_term           = network_settings['ipm_term']
        self.is_treat           = network_settings['is_treat'] #boolean
        self.is_smoothing       = network_settings['is_smoothing'] #boolean
        
        assert self.ipm_term in ['mmd_lin', 'wasserstein', 'no_ipm']

        self.clipping_thres     = 10.

        
        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):
            #### PLACEHOLDER DECLARATION
            self.lr_rate        = tf.placeholder(tf.float32, [], name='learning_rate')
            self.k_prob         = tf.placeholder(tf.float32, [], name='keep_probability')   #keeping rate
            self.alpha          = tf.placeholder(tf.float32, [], name='alpha')
            self.beta           = tf.placeholder(tf.float32, [], name='beta')
            self.gamma          = tf.placeholder(tf.float32, [], name='gamma')

            self.x              = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='inputs')
            self.y              = tf.placeholder(tf.float32, shape=[None, self.num_Event], name='labels')   #event/censoring label (censoring: the last column)
            self.t              = tf.placeholder(tf.float32, shape=[None, 1], name='times')
            self.a              = tf.placeholder(tf.float32, shape=[None, 1], name='treatment_assignments')
            self.w              = tf.placeholder(tf.float32, shape=[None, self.t_max, 2], name='weights')            
            
            self.is_training    = tf.placeholder(tf.bool, name = 'train_test_indicator') #for batch_normalization
            
            self.mb_size        = tf.shape(self.x)[0]
            
            ### mask generation -- for easier computation for at-risk patients
            tmp_range      = tf.cast(tf.expand_dims(tf.range(0, self.t_max, 1), axis=0), tf.float32)
            self.mask1     = tf.cast(tf.equal(tmp_range, self.t), tf.float32)
            self.mask2     = tf.cast(tf.less_equal(tmp_range, self.t), tf.float32)

            y_expanded     = self.mask1 * self.y
            
            
            #PHI(x)
            self.z              = fcnet(
                x_=self.x, o_dim_=self.z_dim, o_fn_=None, 
                num_layers_=self.num_layers1, h_dim_=self.h_dim1, activation_fn=self.active_fn, 
                keep_prob_=self.k_prob, name='encoder'
            )
            
            
            ###BATCH NORMALIZATION. (This follows the implementation of CFRNet)
#             self.z = tf.math.l2_normalize(self.z, axis=0)
            self.z = tf.layers.batch_normalization(self.z, training=self.is_training)
            self.z = self.active_fn(self.z)
            self.z = tf.nn.dropout(self.z, keep_prob=self.k_prob)


            ### H(Z; A,T)
            for m in range(self.t_max):
                tmp_A1              = fcnet(
                    x_=self.z, o_dim_=1, o_fn_=None, 
                    num_layers_=self.num_layers2, h_dim_=self.h_dim2, activation_fn=self.active_fn, 
                    keep_prob_=self.k_prob, name='hypothesis_A1_T{}'.format(m)
                )
                
                if self.is_treat:
                    tmp_A0              = fcnet(
                        x_=self.z, o_dim_=1, o_fn_=None, 
                        num_layers_=self.num_layers2, h_dim_=self.h_dim2, activation_fn=self.active_fn, 
                        keep_prob_=self.k_prob, name='hypothesis_A0_T{}'.format(m)
                    )
                else:
                    tmp_A0              = tf.zeros_like(tmp_A1)
                    
                if m == 0:
                    self.logits_A1 = tmp_A1
                    self.logits_A0 = tmp_A0
                else:
                    self.logits_A1 = tf.concat([self.logits_A1, tmp_A1], axis=1)
                    self.logits_A0 = tf.concat([self.logits_A0, tmp_A0], axis=1)
                    
            
            ### loss - IPM regularization
            self.loss_IPM1 = 0. #treated
            self.loss_IPM0 = 0. #not-treated

            self.w_clipped = tf.clip_by_value(self.w, 0., self.clipping_thres, name='weights_clipped')

            if self.ipm_term != 'no_ipm':
                # for m in range(1, self.t_max):
                for m in range(0, self.t_max):
                    idx1             = tf.where(tf.equal(self.a[:, 0]*self.mask2[:, m], 1.))[:,0]
                     
                    if self.is_treat:
                        idx0             = tf.where(tf.equal((1.-self.a[:, 0])*self.mask2[:, m], 1.))[:,0]
                    
                    
                    if self.ipm_term == 'mmd_lin':
                        self.loss_IPM1 += tf.cond(tf.equal(tf.size(idx1), 0),
                                                  lambda: tf.constant(0, tf.float32),
                                                  lambda: mmd2_lin(self.z, 
                                                                   tf.gather(self.z, idx1, axis=0), 
                                                                   tf.ones_like(self.z[:,0]), 
                                                                   tf.gather(self.w_clipped[:, m, 0], idx1, axis=0))
                                                 )
                        if self.is_treat:
                            self.loss_IPM0 += tf.cond(tf.equal(tf.size(idx0), 0),
                                                      lambda: tf.constant(0, tf.float32),
                                                      lambda: mmd2_lin(self.z, 
                                                                          tf.gather(self.z, idx0, axis=0), 
                                                                          tf.ones_like(self.z[:,0]), 
                                                                          tf.gather(self.w_clipped[:, m, 1], idx0, axis=0))
                                                     )
                        
                    elif self.ipm_term == 'wasserstein':
                        self.loss_IPM1 += tf.cond(tf.equal(tf.size(idx1), 0),
                                                  lambda: tf.constant(0, tf.float32),
                                                  lambda: wasserstein(self.z, 
                                                                      tf.gather(self.z, idx1, axis=0), 
                                                                      tf.ones_like(self.z[:,0]), 
                                                                      tf.gather(self.w_clipped[:, m, 0], idx1, axis=0))
                                                 )
                        if self.is_treat:
                            self.loss_IPM0 += tf.cond(tf.equal(tf.size(idx0), 0),
                                                      lambda: tf.constant(0, tf.float32),
                                                      lambda: wasserstein(self.z, 
                                                                          tf.gather(self.z, idx0, axis=0), 
                                                                          tf.ones_like(self.z[:,0]), 
                                                                          tf.gather(self.w_clipped[:, m, 1], idx0, axis=0))
                                                     )
            self.loss_IPM = self.loss_IPM1 + self.loss_IPM0



            ### loss - smoothing regularization
            self.loss_smoothing_A1 = 0. #treated
            self.loss_smoothing_A0 = 0. #not-treated


            if self.is_smoothing:
                for m in range(1, self.t_max):
                    tmp_Wprev = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/hypothesis_A1_T{}'.format(m-1))[::2]
                    tmp_Wcurr = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/hypothesis_A1_T{}'.format(m))[::2]
                    for l in range(self.num_layers2):
                        self.loss_smoothing_A1 += tf.reduce_mean((tmp_Wprev[l] - tmp_Wcurr[l])**2) ## average over each parameter (for scaling)
                    
                    if self.is_treat:
                        tmp_Wprev = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/hypothesis_A0_T{}'.format(m-1))[::2]
                        tmp_Wcurr = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/hypothesis_A0_T{}'.format(m))[::2]
                        for l in range(self.num_layers2):
                            self.loss_smoothing_A0 += tf.reduce_mean((tmp_Wprev[l] - tmp_Wcurr[l])**2) ## average over each parameter (for scaling)
            
            self.loss_smoothing = self.loss_smoothing_A1 + self.loss_smoothing_A0             
                    
            tmp_w1 = div(self.w[:, :, 0], tf.reduce_sum(self.mask2 * self.a * self.w[:, :, 0], axis=0, keepdims=True) )
            tmp_w1 = self.mask2 * self.a * tmp_w1
            
            if self.is_treat:
                tmp_w0 = div(self.w[:, :, 1], tf.reduce_sum(self.mask2 * (1.- self.a) * self.w[:, :, 1], axis=0, keepdims=True) )
                tmp_w0 = self.mask2 * (1. - self.a) * tmp_w0
                            
            ### loss - factual loss
            self.loss      = 0
            loss_A1        = tf.reduce_sum(
                tmp_w1 * self.mask2 * self.a * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_expanded, logits=self.logits_A1)
            )
            self.loss     += loss_A1
            if self.is_treat:
                loss_A0        = tf.reduce_sum(
                    tmp_w0 * self.mask2 * (1.- self.a) * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_expanded, logits=self.logits_A0)
                )            
                self.loss     += loss_A0            
                   
            ### l2-regularization    
            self.vars_encoder = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/encoder')
            
            if self.reg_scale != 0:
                vars_reg          = [w for w in self.vars_encoder if 'weights' in w.name]
                regularizer       = tf.contrib.layers.l2_regularizer(scale=self.reg_scale, scope=None)
                loss_reg          = tf.contrib.layers.apply_regularization(regularizer, vars_reg)   
            else:
                loss_reg          = 0.


            self.solver       = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss)
            
            self.loss_total   = self.loss + self.beta * self.loss_IPM + self.gamma * self.loss_smoothing + loss_reg
            self.solver_total = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.loss_total)
            
            
            ### batch-normalization operation
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
                      
                    

    def predict_hazard_A1(self, x_):
        odd    = tf.exp(self.logits_A1)
        hazard = odd / (1. + odd)
        return self.sess.run(hazard, feed_dict={self.x:x_, self.k_prob:1.0, self.is_training: False})
         
    def predict_hazard_A0(self, x_):
        odd    = tf.exp(self.logits_A0)
        hazard = odd / (1. + odd)
        return self.sess.run(hazard, feed_dict={self.x:x_, self.k_prob:1.0, self.is_training: False})
    
    def predict_survival_A1(self, x_):
        hazard         = self.predict_hazard_A1(x_)  
        surv           = np.ones_like(hazard)
#         surv[:, 1:]    = np.cumprod(1. - hazard, axis=1)[:, :-1]
        surv[:, :]    = np.cumprod(1. - hazard, axis=1)
        return surv
    
    def predict_survival_A0(self, x_):
        hazard         = self.predict_hazard_A0(x_)  
        surv           = np.ones_like(hazard)
#         surv[:, 1:]    = np.cumprod(1. - hazard, axis=1)[:, :-1]
        surv[:, :]    = np.cumprod(1. - hazard, axis=1)
        return surv
        
        
    def train_baseline(self, x_, y_, t_, a_, lr_train_=1e-3, k_prob_=1.0):
        return self.sess.run([self.solver, self.extra_update_ops, self.loss],
                             feed_dict={self.x:x_, self.y:y_, self.t:t_, self.a:a_, self.w:np.ones([np.shape(x_)[0], self.t_max, 2]),
                                        self.lr_rate:lr_train_, 
                                        self.k_prob:k_prob_,
                                        self.is_training: True})

    def get_loss_basline(self, x_, y_, t_, a_, k_prob_=1.0):
        return self.sess.run(self.loss,
                             feed_dict={self.x:x_, self.y:y_, self.t:t_, self.a:a_, self.w:np.ones([np.shape(x_)[0], self.t_max, 2]),
                                        self.k_prob:k_prob_,
                                        self.is_training: False})
    
    
    def train(self, x_, y_, t_, a_, w_, beta_=1e-3, gamma_=1e-3, lr_train_=1e-3, k_prob_=1.0):
        if not self.is_smoothing:
            gamma_ = 0.
        return self.sess.run([self.solver_total, self.extra_update_ops, self.loss_total, self.loss, self.loss_IPM],
                             feed_dict={self.x:x_, self.y:y_, self.t:t_, self.a:a_, self.w:w_,
                                        self.beta:beta_, self.gamma:gamma_,
                                        self.lr_rate:lr_train_, 
                                        self.k_prob:k_prob_,
                                        self.is_training: True})

    def get_loss(self, x_, y_, t_, a_, w_, beta_=1e-3, gamma_=1e-3, k_prob_=1.0):
        if not self.is_smoothing:
            gamma_ = 0.
        return self.sess.run([self.loss_total, self.loss, self.loss_IPM],
                             feed_dict={self.x:x_, self.y:y_, self.t:t_, self.a:a_, self.w:w_,
                                        self.beta:beta_, self.gamma:gamma_,
                                        self.k_prob:k_prob_,
                                        self.is_training: False})
    
    

