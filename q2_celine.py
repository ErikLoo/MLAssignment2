# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""

#submission for 1000674371 Peiyao Li

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## my code starts here

    test_row = test_datum.reshape(1, test_datum.shape[0])

    calc_dist = - l2(test_row, x_train) / 2 / (tau**2)
    from scipy.special import logsumexp #recommended by instruction
    a = np. exp(calc_dist- logsumexp(calc_dist))
    a = np.diag(a[0])

    x_product = np. dot(np.dot (x_train.transpose(), a), x_train)
    y_product = np.dot(np.dot(x_train.transpose(), a), y_train)

    # print("x product is:", x_product,x_product.shape)
    # print("y product is ", y_product, y_product.shape)

    #solve using the recommended function linalg.solve
    w = np.linalg.solve((x_product + lam*np.identity(x_train.shape[1])), y_product)
    predict = np.dot(w, test_datum)
    #removed return none
    return predict


def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    #my code starts here




    #split test train data
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = val_frac, random_state = 0)

    #calculate loss for each tau
    l_train = np.empty_like(taus)
    l_test = np.empty_like(taus)
    i=0

    for num in taus:
        predict_train = [LRLS(test_datum, x_train, y_train, num) for test_datum in x_train]
        predict_test = [LRLS(test_datum, x_train, y_train, num) for test_datum in x_test]
        predict_train = np.asarray(predict_train)
        predict_test = np.asarray(predict_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        l_train[i] = np.mean((predict_train - y_train)**2)
        l_test [i] = np.mean((predict_test - y_test)**2)
        i+=1
    return l_train, l_test


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,50)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    #plot the loss vs tau
    plt.semilogx(taus, train_losses)
    plt.show()
    plt.semilogx(taus, test_losses)
    plt.show()
    print("finished execution")

