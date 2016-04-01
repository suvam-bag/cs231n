import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_classes = W.shape[0] #gives the no of rows
  num_train = X.shape[1] #gives the end of columns

  
  delta = 1
  for i in xrange(num_train):
    scores = W.dot(X[:, i])

    #normalization in order to avoid instaboil
    #create a temp list to store the exp of the margins
    exp_margin = []
    
  
    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    scores_max = np.max(scores)
    scores -= scores_max
    for j in scores:
      #margin = scores[j] - correct_class_score + delta
      #but for softmax this is where it differs......
      #for softmax exp(scores[j]) and then normalize by add all the scores[j] and then divide each exp(scores[j]) by the sum
      exp_margin.append(np.exp(j))

    #taking the equation from the softmax notes
    loss += -scores[y[i]] + np.log(sum(exp_margin))
    for k in xrange(num_classes):
      margin = exp_margin[k]/(sum(exp_margin))
      #if k == y[i]:
      #the gradient implemntation is bette than the svm  - its essentially the formula of gradient lim(h->0)[(f(x+h) - f(x))/h]
      #a good explanation of the gradient - cs231n.github.io/optimization-1/
      dW[k, :] += (margin-(k == y[i])) * X[:, i]
      #loss += -np.log(exp_margin[k]/sum(exp_margin))
    pass

   # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  print loss
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W) 
  print loss

  #Addin gthe gradient of the regularization to the loss
  dW += reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

  # taken from https://github.com/MyHumbleSelf/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0] #gives the no of rows
  num_train = X.shape[1] #gives the end of columns

  delta = 1

  #computing scores
  scores = np.dot(W,X)

  #normalizing scores
  scores -= np.max(scores)

  #vectorization
  scores_correct = scores[y, range(num_train)]
  loss = -np.mean( np.log(np.exp(scores_correct)/np.sum(np.exp(scores))) )

  #gradient 
  margin = np.exp(scores)/np.sum(np.exp(scores), axis=0)
  ind = np.zeros(margin.shape)
  ind[y, range(num_train)] = 1
  dW = np.dot((margin-ind), X.T)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW /= num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
