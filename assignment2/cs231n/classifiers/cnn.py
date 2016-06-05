import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
  
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
      
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                   hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                   dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.num_layers = 3
        self.use_batchnorm = use_batchnorm
        self.bn_params = {}

        ###############
        ### N, C, H, W = x.shape
        ### F, C, HH, WW = w.shape
        ### F, = b.shape
        ###############
        
        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        #since its a 2x2 max pool, it's divided by half, as the affine layer will expect reduced sizes 
        self.params['W2'] = weight_scale * np.random.randn(num_filters * input_dim[1]/2 * input_dim[2]/2, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if self.use_batchnorm:
            self.bn_params['bn_param1'] = {'mode': 'train',
                                            'running_mean': np.zeros(num_filters),
                                            'running_var': np.zeros(num_filters)}
            self.params['gamma1'] = np.ones(num_filters)
            self.params['beta1'] = np.zeros(num_filters)

            self.bn_params['bn_param2'] = {'mode': 'train',
                                            'running_mean': np.zeros(hidden_dim),
                                            'running_var': np.zeros(hidden_dim)}
            self.params['gamma2'] = np.ones(hidden_dim)
            self.params['beta2'] = np.zeros(hidden_dim)
            
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
     
 
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        if self.use_batchnorm:
            bn_param1, gamma1, beta1 = self.bn_params['bn_param1'], self.params['gamma1'], self.params['beta1']
            bn_param2, gamma2, beta2 = self.bn_params['bn_param2'], self.params['gamma2'], self.params['beta2']
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
     
        if self.use_batchnorm:
            #####conv - norm - relu - 2x2 max pool /- affine norm - relu - /affine - softmax#####
            out_1, cache_1 = conv_norm_relu_pool_forward(X, W1, b1, conv_param, pool_param, gamma1, beta1, bn_param1)
            out_2, cache_2 = affine_relu_batch_norm_forward(out_1, W2, b2, gamma2, beta2, bn_param2)
            out_3, cache_3 = affine_forward(out_2, W3, b3)

        else:
            #####conv - relu - 2x2 max pool /- affine - relu - /affine - softmax#####
            out_1, cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
            out_2, cache_2 = affine_relu_forward(out_1, W2, b2)
            out_3, cache_3 = affine_forward(out_2, W3, b3)

        scores = out_3

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        if y is None:
          return scores
        
        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)
        
        for i in xrange(0, self.num_layers):
            W = 'W' + str(i+1)
            loss += 0.5*self.reg*np.sum(self.params[W]*self.params[W])
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        if self.use_batchnorm:
            dx3, grads['W3'], grads['b3'] = affine_backward(dscores, cache_3)
            dx2, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = affine_relu_batch_norm_backward(dx3, cache_2)
            dx1, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_norm_relu_pool_backward(dx2, cache_1)
        else:
            dx3, grads['W3'], grads['b3'] = affine_backward(dscores, cache_3)
            dx2, grads['W2'], grads['b2'] = affine_relu_backward(dx3, cache_2)
            dx1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx2, cache_1)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        return loss, grads
    

class DeepConvNet(object):
    """
    architecture ---
    [conv-relu-pool]xL - [affine]xM - [softmax or SVM]

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.

    """
    def __init__(self, input_dim=(3, 32, 32), conv_filters=[32, 64, 128], filter_size=7,
               hidden_dims=[500, 500], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.bn_params = {}
        self.filter_size = filter_size
        self.num_convLayers = len(conv_filters)
        self.num_affineLayers = len(hidden_dims)
        self.cache = {}

        
        '''
        Hc = (H + 2 * P - filter_size) / stride_conv + 1
        Wc = (W + 2 * P - filter_size) / stride_conv + 1
        width_pool = 2
        height_pool = 2
        stride_pool = 2
        Hp = (Hc - height_pool) / stride_pool + 1
        Wp = (Wc - width_pool) / stride_pool + 1

        
        '''

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (filter_size-1)/2}
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        C = input_dim[0]
        H = input_dim[1]
        W = input_dim[2]
        
        P = conv_param['pad']
        stride_conv = conv_param['stride']
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride_pool = pool_param['stride']

        #conv_layers x L
        dims_conv = [input_dim[0]] + conv_filters
        for i in xrange(self.num_convLayers):
            Hc = (H + 2 * P - filter_size) / stride_conv + 1
            Wc = (W + 2 * P - filter_size) / stride_conv + 1
            H_out = (Hc - pool_height) / stride_pool + 1
            W_out = (Wc - pool_width) / stride_pool + 1
            index = str(i+1) 
            W = 'W' + index
            b = 'b' + index
            self.params[W] = weight_scale * np.random.randn(dims_conv[i+1], dims_conv[i], filter_size, filter_size)   #(K, D, F, F)
            self.params[b] = np.zeros(dims_conv[i+1])
            H = H_out
            W = W_out
            if self.use_batchnorm:
                bn_param = 'bn_param' + index
                gamma = 'gamma' + index
                beta = 'beta' + index
                self.bn_params[bn_param] = {'mode': 'train',
                                            'running_mean': np.zeros(dims_conv[i+1]),
                                            'running_var': np.zeros(dims_conv[i+1])}
                self.params[gamma] = np.ones(dims_conv[i+1])
                self.params[beta] = np.zeros(dims_conv[i+1])
        

        #affine_layers x M
        dims = [H_out * W_out * dims_conv[-1]] + hidden_dims
        for i in xrange(self.num_affineLayers):
            idx = i + self.num_convLayers + 1
            index_affine = str(idx)
            W = 'W' + index_affine
            b = 'b' + index_affine
            self.params[W] = weight_scale * np.random.randn(dims[i], dims[i+1])
            self.params[b] = np.zeros(dims[i+1])
            if self.use_batchnorm:
                bn_param = 'bn_param' + index_affine
                gamma = 'gamma' + index_affine
                beta = 'beta' + index_affine
                self.bn_params[bn_param] = {'mode': 'train',
                                            'running_mean': np.zeros(dims[i+1]),
                                            'running_var': np.zeros(dims[i+1])}
                self.params[gamma] = np.ones(dims[i+1])
                self.params[beta] = np.zeros(dims[i+1])


        #scoring layer
        idx_score = self.num_convLayers + self.num_affineLayers + 1
        W_score = 'W' + str(idx_score)
        b_score = 'b' + str(idx_score)  
        self.params[W_score] = weight_scale * np.random.randn(dims[-1], num_classes)
        self.params[b_score] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        out = X
        for i in xrange(self.num_convLayers):
            index = str(i+1)
            W = 'W' + index
            b = 'b' + index
            cache_name = 'cache_' + index
            W, b = self.params[W], self.params[b]
            if self.use_batchnorm:
                bn_param = 'bn_param' + index
                gamma = 'gamma' + index
                beta = 'beta' + index
                bn_param, gamma, beta = self.bn_params[bn_param], self.params[gamma], self.params[beta]
                out, cache = conv_norm_relu_pool_forward(out, W, b, conv_param, pool_param, gamma, beta, bn_param)
            else:
                out, cache = conv_relu_pool_forward(out, W, b, conv_param, pool_param)
            self.cache[cache_name] = cache


        for i in xrange(self.num_affineLayers):
            idx = i + self.num_convLayers + 1
            index_affine = str(idx)
            W = 'W' + index_affine
            b = 'b' + index_affine
            cache_name = 'cache_' + index_affine
            W, b = self.params[W], self.params[b]
            if self.use_batchnorm:
                bn_param = 'bn_param' + index_affine
                gamma = 'gamma' + index_affine
                beta = 'beta' + index_affine
                bn_param, gamma, beta = self.bn_params[bn_param], self.params[gamma], self.params[beta]
                out, cache = affine_relu_batch_norm_forward(out, W, b, gamma, beta, bn_param)
            else:
                out, cache = affine_relu_forward(out, W, b)
            self.cache[cache_name] = cache


        idx_score = self.num_convLayers + self.num_affineLayers + 1
        W_score = 'W' + str(idx_score)
        b_score = 'b' + str(idx_score)
        cache_score = 'cache' + str(idx_score)
        W, b = self.params[W_score], self.params[b_score]
        scores, cache_score = affine_forward(out, W, b)

        if y is None:
            return scores
    
        loss, grads = 0, {}

        #loss, dscores = softmax_loss(scores, y)
        loss, dscores = svm_loss(scores, y)
        
        for i in xrange(self.num_convLayers + self.num_affineLayers + 1):
            W = 'W' + str(i+1)
            loss += 0.5*self.reg*np.sum(self.params[W]*self.params[W])
        
        dx_affine, grads[W_score], grads[b_score] = affine_backward(dscores, cache_score)
        
        for i in xrange((self.num_affineLayers), 0, -1):
            idx = i + self.num_convLayers 
            index_affine = str(idx)
            W = 'W' + index_affine
            b = 'b' + index_affine
            cache_name = 'cache_' + index_affine
            if self.use_batchnorm:
                bn_param = 'bn_param' + index_affine
                gamma = 'gamma' + index_affine
                beta = 'beta' + index_affine
                dx_affine, grads[W], grads[b], grads[gamma], grads[beta] = affine_relu_batch_norm_backward(dx_affine, self.cache[cache_name])
            else:
                dx_affine, grads[W], grads[b] = affine_relu_backward(dx_affine, self.cache[cache_name])


        dx_conv = dx_affine
        for i in xrange((self.num_convLayers), 0, -1):
            index = str(i)
            W = 'W' + index
            b = 'b' + index
            cache_name = 'cache_' + index
            if self.use_batchnorm:
                bn_param = 'bn_param' + index
                gamma = 'gamma' + index
                beta = 'beta' + index
                dx_conv, grads[W], grads[b], grads[gamma], grads[beta] = conv_norm_relu_pool_backward(dx_conv, self.cache[cache_name])
            else:
                dx_conv, grads[W], grads[b] = conv_relu_pool_backward(dx_conv, self.cache[cache_name])
            pass
 
    
        return loss, grads
    
pass
        

class DeepConvNet2(object):
    """
    architecture ---
    [conv-relu-pool]xL - conv - relu - [affine]xM - [softmax or SVM]

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.

    """
    def __init__(self, input_dim=(3, 32, 32), conv_filters=[32, 64, 128], conv_intermediate_filter=128, filter_size=7,
               hidden_dims=[500, 500], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.bn_params = {}
        self.filter_size = filter_size
        self.num_convLayers = len(conv_filters)
        self.num_affineLayers = len(hidden_dims)
        self.cache = {}

        
        '''
        Hc = (H + 2 * P - filter_size) / stride_conv + 1
        Wc = (W + 2 * P - filter_size) / stride_conv + 1
        width_pool = 2
        height_pool = 2
        stride_pool = 2
        Hp = (Hc - height_pool) / stride_pool + 1
        Wp = (Wc - width_pool) / stride_pool + 1

        
        '''

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (filter_size-1)/2}
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        C = input_dim[0]
        H = input_dim[1]
        W = input_dim[2]
        
        P = conv_param['pad']
        stride_conv = conv_param['stride']
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride_pool = pool_param['stride']

        #conv_layers x L
        dims_conv = [input_dim[0]] + conv_filters
        for i in xrange(self.num_convLayers):
            Hc = (H + 2 * P - filter_size) / stride_conv + 1
            Wc = (W + 2 * P - filter_size) / stride_conv + 1
            H_out = (Hc - pool_height) / stride_pool + 1
            W_out = (Wc - pool_width) / stride_pool + 1
            index = str(i+1) 
            W = 'W' + index
            b = 'b' + index
            self.params[W] = weight_scale * np.random.randn(dims_conv[i+1], dims_conv[i], filter_size, filter_size)   #(K, D, F, F)
            self.params[b] = np.zeros(dims_conv[i+1])
            H = H_out
            W = W_out
            if self.use_batchnorm:
                bn_param = 'bn_param' + index
                gamma = 'gamma' + index
                beta = 'beta' + index
                self.bn_params[bn_param] = {'mode': 'train',
                                            'running_mean': np.zeros(dims_conv[i+1]),
                                            'running_var': np.zeros(dims_conv[i+1])}
                self.params[gamma] = np.ones(dims_conv[i+1])
                self.params[beta] = np.zeros(dims_conv[i+1])
        

        #conv - relu layer
        index_conv_intermediate = str(self.num_convLayers + 1)
        W = 'W' + index_conv_intermediate
        b = 'b' + index_conv_intermediate
        H_out = (H_out + 2 * P - filter_size) / stride_conv + 1
        W_out = (W_out + 2 * P - filter_size) / stride_conv + 1
        self.params[W] = weight_scale * np.random.randn(conv_intermediate_filter, conv_filters[-1], filter_size, filter_size)
        self.params[b] = np.zeros(conv_intermediate_filter)
        if self.use_batchnorm:
            bn_param = 'bn_param' + index_conv_intermediate
            gamma = 'gamma' + index_conv_intermediate
            beta = 'beta' + index_conv_intermediate
            self.bn_params[bn_param] = {'mode': 'train',
                                        'running_mean': np.zeros(conv_intermediate_filter),
                                        'running_var': np.zeros(conv_intermediate_filter)}
            self.params[gamma] = np.ones(conv_intermediate_filter)
            self.params[beta] = np.zeros(conv_intermediate_filter)


        #affine_layers x M
        dims = [H_out * W_out * conv_intermediate_filter] + hidden_dims
        for i in xrange(self.num_affineLayers):
            idx = i + self.num_convLayers + 2
            index_affine = str(idx)
            W = 'W' + index_affine
            b = 'b' + index_affine
            self.params[W] = weight_scale * np.random.randn(dims[i], dims[i+1])
            self.params[b] = np.zeros(dims[i+1])
            if self.use_batchnorm:
                bn_param = 'bn_param' + index_affine
                gamma = 'gamma' + index_affine
                beta = 'beta' + index_affine
                self.bn_params[bn_param] = {'mode': 'train',
                                            'running_mean': np.zeros(dims[i+1]),
                                            'running_var': np.zeros(dims[i+1])}
                self.params[gamma] = np.ones(dims[i+1])
                self.params[beta] = np.zeros(dims[i+1])


        #scoring layer
        idx_score = self.num_convLayers + self.num_affineLayers + 2
        W_score = 'W' + str(idx_score)
        b_score = 'b' + str(idx_score)  
        self.params[W_score] = weight_scale * np.random.randn(dims[-1], num_classes)
        self.params[b_score] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_batchnorm:
            for key, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        out = X
        for i in xrange(self.num_convLayers):
            index = str(i+1)
            W = 'W' + index
            b = 'b' + index
            cache_name = 'cache_' + index
            W, b = self.params[W], self.params[b]
            if self.use_batchnorm:
                bn_param = 'bn_param' + index
                gamma = 'gamma' + index
                beta = 'beta' + index
                bn_param, gamma, beta = self.bn_params[bn_param], self.params[gamma], self.params[beta]
                out, cache = conv_norm_relu_pool_forward(out, W, b, conv_param, pool_param, gamma, beta, bn_param)
            else:
                out, cache = conv_relu_pool_forward(out, W, b, conv_param, pool_param)
            self.cache[cache_name] = cache


        #conv - relu layer
        index_conv_intermediate = str(self.num_convLayers + 1)
        W = 'W' + index_conv_intermediate
        b = 'b' + index_conv_intermediate
        cache_name = 'cache_' + index_conv_intermediate
        W, b = self.params[W], self.params[b]
        if self.use_batchnorm:
            bn_param = 'bn_param' + index_conv_intermediate
            gamma = 'gamma' + index_conv_intermediate
            beta = 'beta' + index_conv_intermediate
            bn_param, gamma, beta = self.bn_params[bn_param], self.params[gamma], self.params[beta]
            out, cache = conv_norm_relu_forward(out, W, b, conv_param, gamma, beta, bn_param)
        else:
            out, cache = conv_relu_forward(out, W, b, conv_param)
        self.cache[cache_name] = cache


        for i in xrange(self.num_affineLayers):
            idx = i + self.num_convLayers + 2
            index_affine = str(idx)
            W = 'W' + index_affine
            b = 'b' + index_affine
            cache_name = 'cache_' + index_affine
            W, b = self.params[W], self.params[b]
            if self.use_batchnorm:
                bn_param = 'bn_param' + index_affine
                gamma = 'gamma' + index_affine
                beta = 'beta' + index_affine
                bn_param, gamma, beta = self.bn_params[bn_param], self.params[gamma], self.params[beta]
                out, cache = affine_relu_batch_norm_forward(out, W, b, gamma, beta, bn_param)
            else:
                out, cache = affine_relu_forward(out, W, b)
            self.cache[cache_name] = cache


        idx_score = self.num_convLayers + self.num_affineLayers + 2
        W_score = 'W' + str(idx_score)
        b_score = 'b' + str(idx_score)
        cache_score = 'cache' + str(idx_score)
        W, b = self.params[W_score], self.params[b_score]
        scores, cache_score = affine_forward(out, W, b)

        if y is None:
            return scores
    
        loss, grads = 0, {}

        #loss, dscores = softmax_loss(scores, y)
        loss, dscores = softmax_loss(scores, y)
        
        for i in xrange(self.num_convLayers + self.num_affineLayers + 2):
            W = 'W' + str(i+1)
            loss += 0.5*self.reg*np.sum(self.params[W]*self.params[W])
        

        dx_affine, grads[W_score], grads[b_score] = affine_backward(dscores, cache_score)

        
        for i in xrange((self.num_affineLayers), 0, -1):
            idx = i + self.num_convLayers + 1
            index_affine = str(idx)
            W = 'W' + index_affine
            b = 'b' + index_affine
            cache_name = 'cache_' + index_affine
            if self.use_batchnorm:
                bn_param = 'bn_param' + index_affine
                gamma = 'gamma' + index_affine
                beta = 'beta' + index_affine
                dx_affine_relu, grads[W], grads[b], grads[gamma], grads[beta] = affine_relu_batch_norm_backward(dx_affine, self.cache[cache_name])
            else:
                dx_affine_relu, grads[W], grads[b] = affine_relu_backward(dx_affine, self.cache[cache_name])


        index_conv_intermediate = str(self.num_convLayers + 1)
        W = 'W' + index_conv_intermediate
        b = 'b' + index_conv_intermediate
        cache_name = 'cache_' + index_conv_intermediate
        if self.use_batchnorm:
            bn_param = 'bn_param' + index_conv_intermediate
            gamma = 'gamma' + index_conv_intermediate
            beta = 'beta' + index_conv_intermediate
            dx_conv_intermediate, grads[W], grads[b], grads[gamma], grads[beta] = conv_norm_relu_backward(dx_affine_relu, self.cache[cache_name])
        else:
            dx_conv_intermediate, grads[W], grads[b] = conv_relu_backward(dx_affine_relu, self.cache[cache_name])


        dx_conv = dx_conv_intermediate
        for i in xrange((self.num_convLayers), 0, -1):
            index = str(i)
            W = 'W' + index
            b = 'b' + index
            cache_name = 'cache_' + index
            if self.use_batchnorm:
                bn_param = 'bn_param' + index
                gamma = 'gamma' + index
                beta = 'beta' + index
                dx_conv, grads[W], grads[b], grads[gamma], grads[beta] = conv_norm_relu_pool_backward(dx_conv, self.cache[cache_name])
            else:
                dx_conv, grads[W], grads[b] = conv_relu_pool_backward(dx_conv, self.cache[cache_name])
            pass
 
    
        return loss, grads
    
pass 
    

