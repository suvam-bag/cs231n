import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def ReLU(self, X):
    #return np.maximum(X, 0)
    #or
    #alternative faster way
    return (X * (X > 0))

  def sigmoid(self, X):
    return 1/(1+np.exp(-X))

  def LeakyReLU(self, X, alpha=0.1):
    return (X < 0)*(alpha * X) + (X>=0)*(X)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    #since this is a 2 layer NN model, there is a single hidden layer
    #pseudo code in steps
    #  1. f = activation function..............ReLU
    #  2. x = input vector ....................dimension = (N, D).......N=5,D=4
    #.....W1 = DxH....H - hidden layer dimension
    #.....W2 = HxC....C - no of classes
    #  3. h = f(np.dot(w1,x) + b1).............dimension = (D, H).......H=10,D=4
    #  4. scores = np.dot(w2,h) + b2...........dimension = (N, C).......N=5,C=3
    
    

    





    #X.shape[1] - the number of columns of the input data i.e - input size 
    #for iter in xrange(10000):
      #forthe scores  - the only difference with the softmax classifier is that NN 
      #                 has a hidden layer and an activation function
    
    # Compute the loss
    loss = 0.0

    #iterate through each input
    #number of inputs is N = X.shape[0]
    #for x in xrange(X.shape[0]):

    #num_inputs = X.shape[0]    #layer 0
    #class scores (NxC)
    #first layer - the hidden layer
    h = self.ReLU(np.dot(X,W1) + b1)   #layer 1
    #the final layer - output
    scores = (np.dot(h,W2) + b2)  #layer 2

    

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    #compute the class probabilities (NxC)
    exp_scores = np.exp(scores)
    #probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    scores = (exp_scores.T/np.sum(exp_scores, axis=1)).T

    #softmax loss
    #y - labels
    correct_logprobs = -np.log(scores[range(N), y])
    #data loss
    loss = np.sum(correct_logprobs)/N    
    #total loss = data loss + reg loss
    #reg - regularization factor provided by jupyter notebook
    loss += 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################


    
    #step1 - gradient on scores - grads['scores']
    #dscores = probs #taking the normal score instead of the cross-entropy softmax
    #taken from the 13 line NN code
    #def sigmoid_out[put_to_derivative(output):
    #  return output*(1-output)  
    #dscores[range(N),y] -= 1     
    #dscores /= N
    #grads['scores'] = dscores


    Y = np.zeros(scores.shape)
    for i,row in enumerate(Y):
      row[y[i]] = 1

    dscores = (scores - Y)

    #step2 - evaulate the gradients on W2 and b2 - grads['W2'] & grads['b2']
    #backproping into parameters W2 and b2
    dW2 = np.dot((dscores.T),h).T/N  #taken from the 13 line 2 layer NN - synapse_0_derivative = np.dot(layer_0.T,layer_1_delta)
    dW2 += reg * W2       #adding the regularization factor to dW2
    grads['W2'] = dW2
    #grads['b2'] = np.sum(dscores, axis=0, keepdims=True)   #biases dont have regularizartion terms
    grads['b2'] = np.dot((dscores.T),(np.ones(h.shape[0])))/N


    #step3 - evaluate the gradient on the hidden layer - grads['h']
    #backprop into the hidden layer
    dh = np.dot(dscores, W2.T)
    dh[h <= 0] = 0 #essentially eliminate <=0 ...ReLU 
    #grads['h'] = dh

    #step4 - evaluate the gradients on W1 snd b1 - grads['W1'] & grads['b1']
    #backprop into W1 and b1  
    #dW1 = np.dot(X.T, dh)      
    dW1 = np.dot((dh.T),X).T/N
    dW1 += reg * W1        #adding the regularization factor to dW1
    grads['W1'] = dW1 
    #grads['b1'] = np.sum(dh, axis=0, keepdims=True)   #biases dont have regularization terms
    grads['b1'] = np.dot((dh.T),(np.ones(X.shape[0]))).T/N





    '''
    
    X1 = np.maximum( X.dot(W1) + b1, 0)   #this is essentially h
    X2 = X1.dot(W2) + b2    #this is essentially scores

    scores = X2

    Y = np.zeros(X2.shape)
    for i,row in enumerate(Y):
      row[y[i]] = 1

    dX2 = scores - Y
    dW2 = ((dX2.T).dot(X1)).T/N + reg*W2
    db2 = (dX2.T).dot(np.ones(X1.shape[0]))/N
    dX1 = dX2.dot(W2.T)
    dW1 = (((dX1*(X1>0)).T).dot(X)).T/N + reg*W1
    db1 = (((dX1*(X1>0)).T).dot(np.ones(X.shape[0]))).T/N
      
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1
     
    '''


    '''
    #refere to http://iamtrask.github.io/2015/07/27/python-network-part2/

    Y = np.zeros(scores.shape)
    #taking the derivative of the ReLU for scores
    #if X>0, dReLU = 1 else dReLU = 0
    for i,row in enumerate(Y):
      row[y[i]] = 1

    
    dX2 = scores - Y                                            #loss for the second layer i.e - error
    dW2 = ((dX2.T).dot(h)).T/N + reg*W2                         #gradient on W2 
    db2 = (dX2.T).dot(np.ones(h.shape[0]))/N                    #gradient on b2
    dh = dX2.dot(W2.T)                                          #gradient on hidden layer
    dW1 = (((dh*(h>0)).T).dot(X)).T/N + reg*W1                  #gradient on W1
    db1 = (((dh*(h>0)).T).dot(np.ones(X.shape[0]))).T/N         #gradient on b1

    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1
    '''
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads
    '''
  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):

    '''

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=0.5, num_iters=1000,
            batch_size=200, verbose=True):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """

    

    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_indices = np.random.choice(num_train, batch_size)
      
      X_batch = X[batch_indices]
  
      y_batch = y[batch_indices]

      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)

      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################

      
      #parameters update (vanilla update)
      # Unpack variables from the params dictionary
      #W1, b1 = self.params['W1'], self.params['b1']
      #W2, b2 = self.params['W2'], self.params['b2']

      #using the gradients stored in the grads dictionary
      
      self.params['W1'] -= learning_rate * grads['W1']
      #print self.params['W1'].shape
      #print grads['W1'].shape
      self.params['b1'] -= learning_rate * grads['b1']
      #print self.params['b1'].shape
      #print grads['b1'].shape
      self.params['W2'] -= learning_rate * grads['W2']
      #print self.params['W2'].shape
      #print grads['W2'].shape
      self.params['b2'] -= learning_rate * grads['b2']
      #print self.params['b2'].shape
      #print grads['b1'].shape
      
      '''
      for variable in self.params:
        self.params[variable] += -learning_rate*grads[variable]
        print self.params[variable].shape
        print grads[variable].shape
      '''
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.
    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.
    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    #h = np.maximum( X.dot(self.params['W1']) + self.params['b1'], 0 )
    h = self.ReLU(X.dot(self.params['W1']) + self.params['b1'])
    #X2 = h.dot(self.params['W2']) + self.params['b2']
    X2 = np.dot(h, self.params['W2']) + self.params['b2']
    exp_X2 = np.exp(X2)
    scores = (exp_X2.T/np.sum(exp_X2, axis = 1)).T
    
    y_pred = np.argmax(scores, axis = 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################
    return y_pred
