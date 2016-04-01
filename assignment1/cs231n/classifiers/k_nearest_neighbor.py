import numpy as np
#from collections import defaultdict

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = y
    #print ('training done')
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Input:
    X - A num_test x dimension array where each row is a test point.
    k - The number of nearest neighbors that vote for predicted label
    num_loops - Determines which method to use to compute distances
                between training points and test points.

    Output:
    y - A vector of length num_test, where y[i] is the predicted label for the
        test point X[i, :].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Input:
    X - An num_test x dimension array where each row is a test point.

    Output:
    dists - A num_test x num_train array where dists[i, j] is the distance
            between the ith test point and the jth training point.
    """
    num_test = X.shape[0] #number of rows for the test matrix is 500 i.e - 500 samples
    num_train = self.X_train.shape[0] #number of rows for the train matrix is 5000 i.e - 5000 samples
    dists = np.zeros((num_test, num_train)) #size of the dists matrix is 500(rows)x5000(cols)
    #print ('in the dist calculation fucntion')		
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]               #
        #####################################################################
	#each row of both train and test data is a single point
	#hence add all the points in a single row of both matrices resulting in single column matrices
	#hence finally self.X_train[col,0] - X[row,0] or vice versa
	#dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j][i] - X[i][j])))
 	#X[i].sum()
	#self.X_train[j].sum()
	#dists[i, j] = self.X_train[j,0] - X[i,0]
        dists[i,j] = np.sqrt(np.sum(np.square(X[i,:] - self.X_train[j,:])))
        pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
     # X[i].sum()
     # self.X_train[:].sum()
     # dists[i,:] = self.X_train[:,0] - X[i,0]
      dists[i,:] = np.sqrt(np.sum(np.square(X[i,:] - self.X_train), axis=1))

      pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #X[:].sum()
    #self.X_train[:].sum()
    #dists = self.X_train[:,0] - X[:,0]
    
    tDot = np.multiply(np.dot(X, self.X_train.T), -2)
    t1 = np.sum(np.square(X), axis=1, keepdims=True)
    t2 = np.sum(np.square(self.X_train), axis=1)
    tDot = np.add(t1, tDot)
    tDot = np.add(tDot, t2)
    dists = np.sqrt(tDot)
    
    '''
    test_sum = np.sum(np.square(X), axis=1) # num_test x 1
    train_sum = np.sum(np.square(self.X_train), axis=1) # num_train x 1
    inner_product = np.dot(X, self.X_train.T) # num_test x num_train
    dists = np.sqrt(-2 * inner_product + test_sum + train_sum.T) # broadcast!
    '''
    pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Input:
    dists - A num_test x num_train array where dists[i, j] gives the distance
            between the ith test point and the jth training point.

    Output:
    y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
    """
    num_test = dists.shape[0]  #the no of rows of the matrix dist[i,j]....
    y_pred = np.zeros(num_test)  
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # training point, and use self.y_train to find the labels of these      #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################

      #sort the dist[i,j] ...index_y contains the sorted indexes	

      #solution1 
      #########index_y = np.argsort(dists[i,:], axis=None, kind='quicksort', order=None)#######
      #print(len(index_y))
      #print(index_y)
      #closest_y[i] = [label1, label2, ......].......this array contains the label corresponding to the index_y[i]
      #for j in xrange(len(index_y)):
      #	closest_y.append(self.y_train[j])
      #print(closest_y)

      #solution2
      y_indicies = np.argsort(dists[i, :], axis = 0)
      closest_y = self.y_train[y_indicies[:k]]
    
      pass
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #closest_y contains the labels
      
     # d = defaultdict()
     # for j in closest_y:
     # d[j] += 1
     # result = max(d.iterations(), key=lambda x: x[1])
     # y_pred[j] = result[0]

        ###solution 1 but requires closest_y to be a list#####
     # y_pred[i] = max(set(closest_y), key=closest_y.count)

     ###solution2#####
      y_pred[i] = np.argmax(np.bincount(closest_y))
      pass
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

