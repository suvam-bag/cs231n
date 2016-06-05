from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_relu_batch_norm_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU and batch_norm

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """

  a, fc_cache = affine_forward(x, w, b)
  batch_out, batch_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(batch_out)
  cache = (fc_cache, batch_cache, relu_cache)
  return out, cache
pass


def affine_relu_batch_norm_backward(dout, cache):
  """
  Backward pass for the affine-relu-batch_norm convenience layer
  """
  fc_cache, batch_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dbatch_out, dgamma, dbeta = batchnorm_backward(da, batch_cache)
  dx, dw, db = affine_backward(dbatch_out, fc_cache)
  return dx, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def conv_norm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, batch_norm and a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """

  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  batch_norm, batch_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(batch_norm)
  cache = (conv_cache, batch_cache, relu_cache)
  return out, cache

def conv_norm_relu_backward(dout, cache):
  """
  Backward pass for the conv-norm-relu convenience layer
  """
  conv_cache, batch_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dbatch_out, dgamma, dbeta = spatial_batchnorm_backward(da, batch_cache)
  dx, dw, db = conv_backward_fast(dbatch_out, conv_cache)
  return dx, dw, db, dgamma, dbeta

def conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, batch_norm and a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """

  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  batch_norm, batch_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(batch_norm)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, batch_cache, relu_cache, pool_cache)
  return out, cache

def conv_norm_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-norm-relu-pool convenience layer
  """
  conv_cache, batch_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  dbatch_norm = relu_backward(ds, relu_cache)
  dbatch_out, dgamma, dbeta = spatial_batchnorm_backward(dbatch_norm, batch_cache)
  dx, dw, db = conv_backward_fast(dbatch_out, conv_cache)
  return dx, dw, db, dgamma, dbeta


