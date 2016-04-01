import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  #help from misaka-10032.github.io/2015/02/01/cs231n-assignment-1/
  #to understand better how the los is computed look up - cs231n.github.io/optimization-1/
  num_classes = W.shape[0]  # gives the number of rows
  num_train = X.shape[1]    # gives the number of columns
  loss = 0.0
  delta = 1
  for i in xrange(num_train):
    ddW = np.zeros(W.shape)
    ddWyi = np.zeros(W[0].shape)

    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      #from cs231n notes
      margin = scores[j] - correct_class_score + delta 
      #condition based on cs231n notes
      if margin > 0:
        loss += margin
        ddW[j] = X[:, i] ## be careful, it's a reference
        ddWyi += ddW[j]
    ddW[y[i]] = -ddWyi
    dW += ddW
    pass
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss. // misaka-10032.github.io/2015/02/01/cs231n-assignment-1/
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  #Addin gthe gradient of the regularization to the loss
  dW += reg*W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = W.dot(X)
  delta = 1
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # loss
  # correct_class_score shape is (#samples) and it has the scores of right labels
  correct_class_score = [scores[y[i], i] for i in xrange(scores.shape[1])]
  

  margin = scores - correct_class_score + delta
  for i in xrange(scores.shape[1]):
    margin[y[i], i] = 0 ###margin 0 on y[i]
   
  

  # mass is a matrix holding all useful temp results 
  # shape of margin is (#class, #train)
  mass = np.maximum(0, margin)

  loss = np.sum(mass) / X.shape[1]
  loss += 0.5 * reg * np.sum(W * W)

  
  
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # continue on last snippet
  # weight to be producted by X
  # its shape is (#classes, #samples)
  weight = np.array((margin > 0).astype(int))
  
  # weights on y[i] needs special care
  weight_yi = -np.sum(weight, axis=0)
  for i in xrange(scores.shape[1]):
    weight[y[i], i] = weight_yi[i]
  
  # half vectorized
  for i in xrange(X.shape[1]):
    ddW = X[:, i] * weight[:, i].reshape(-1, 1)
    dW += ddW
    
  dW /= X.shape[1]
  dW += reg * W

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
