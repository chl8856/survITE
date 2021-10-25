import tensorflow as tf
import numpy as np

_EPSILON = 1e-08


################################
##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + _EPSILON)

def div(x, y):
    return tf.div(x, (y + _EPSILON))

################################
##### IPM TERMS
def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


# def mmd2_lin(X1,X2,p=0.5):
#     ''' Linear MMD '''
#     mean1 = tf.reduce_mean(X1,reduction_indices=0)
#     mean2 = tf.reduce_mean(X2,reduction_indices=0)

#     mmd = tf.reduce_sum(tf.square(2.0*p*mean1 - 2.0*(1.0-p)*mean2))

#     return mmd

# def mmd2_rbf(X1,X2,p=0.5,sig=0.1):
#     """ Computes the l2-RBF MMD for X1 vs X2 """
#     K11 = tf.exp(-pdist2sq(X1,X1)/tf.square(sig))
#     K12 = tf.exp(-pdist2sq(X1,X2)/tf.square(sig))
#     K22 = tf.exp(-pdist2sq(X2,X2)/tf.square(sig))

#     m = tf.to_float(tf.shape(X1)[0])
#     n = tf.to_float(tf.shape(X2)[0])

#     mmd = tf.square(1.0-p)/(m*(m-1.0))*(tf.reduce_sum(K11)-m)
#     mmd = mmd + tf.square(p)/(n*(n-1.0))*(tf.reduce_sum(K22)-n)
#     mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.reduce_sum(K12)
#     mmd = 4.0*mmd

#     return mmd

def mmd2_lin(X1,X2,W1=None,W2=None,p=0.5,weights=None):
    ''' Linear MMD '''    
    if (W1 is None) and (W2 is None):
        W1 = tf.ones_like(X1[:,0])
        W2 = tf.ones_like(X2[:,0])
    
    W1     = div(W1, tf.reduce_sum(W1))
    W2     = div(W2, tf.reduce_sum(W2))
    
    W1     = tf.reshape(W1, [-1,1])
    W2     = tf.reshape(W2, [-1,1])
        
    mean1 = tf.reduce_sum(W1*X1, axis=0)
    mean2 = tf.reduce_sum(W2*X2, axis=0)
    
    mmd = tf.reduce_sum(tf.square(2.0*p*mean1 - 2.0*(1.0-p)*mean2))
    
    return mmd


def wasserstein(X1,X2,W1=None,W2=None,p=0.5,lam=10,its=10): #,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """    
    n1 = tf.to_float(tf.shape(X1)[0])
    n2 = tf.to_float(tf.shape(X2)[0])
    
    ''' Compute distance matrix'''
    M = pdist2sq(X1,X2)
        
    #for now consider W1 and W2 is [None,] shape
    if (W1 is None) and (W2 is None):
        W1 = tf.ones_like(X1[:,0])
        W2 = tf.ones_like(X2[:,0])
    
    W1     = div(W1, tf.reduce_sum(W1))
    W2     = div(W2, tf.reduce_sum(W2))
    
    W1     = tf.reshape(W1, [-1,1])
    W2     = tf.reshape(W2, [-1,1])
    W_mask = tf.tile(W1, [1, n2]) * tf.tile(tf.transpose(W2), [n1, 1])
    
    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_sum(M*W_mask) #this becomes weighted average
    
    M_drop  = tf.nn.dropout(M, 10/(n1*n2))
    delta   = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt  = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))], axis=0)
    Mt  = tf.concat([M,row], axis=0)
    Mt  = tf.concat([Mt,col], axis=1)

    ''' Compute marginal vectors '''        
    a = tf.concat([p*tf.ones_like(X1[:,0:1])*W1, (1-p)*tf.ones((1,1))], axis=0)
    b = tf.concat([(1-p)*tf.ones_like(X2[:,0:1])*W2, p*tf.ones((1,1))], axis=0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D #, Mlam
