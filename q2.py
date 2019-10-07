# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    #generate A from x_train and test_datum
    #change test_datum into a row vector
    x_test_row = test_datum.reshape(1,test_datum.shape[0])
    distance = l2(x_test_row,x_train)
    #considering exp(x)/sum(exp(x)) = exp(log(exp(x)/sum(exp(x))))
    # = exp(log(exp(x))-log(sum(exp(x))))
    # = exp(x-log(sum(exp(x))))
    exponent = -distance/(2*(tau**2))
    #generate A
    A = np.exp(exponent-logsumexp(exponent))

    #convert A into a diagonal matrix
    A_diag = np.diag(A[0])

    # plug the formula for weights and calculate weights
    #use a linear solver to solve the problem
    #note the weigths is a rwow vector
    w = np.linalg.solve(np.dot(x_train.T,np.dot(A_diag,x_train))+lam*np.identity(x_train.shape[1]),np.dot(x_train.T,np.dot(A_diag,y_train)))

    # Code for verificatio
    # X, Y = x_train, y_train
    # X_T = x_train.transpose()
    # # The two sides of the equation
    # # (use element wise multiplication instead of
    # # matrix multiplication because A is a vector)
    # left = ((X_T * A) @ X) + lam
    # right = (X_T * A) @ Y
    #
    # # Solve (don't use slow inverse)
    # try:
    #     w = np.linalg.solve(left, right)
    # except:
    #     w = np.linalg.pinv(left) @ right
    # code for verification ends

    return np.dot(test_datum,w)



def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO

    #randomly split the data into 70% training and 30% validation
    x_train, x_val, y_train, y_val = train_test_split(x, y.reshape(len(y),1), test_size=0.3)

    loss_train = np.array([])
    loss_val = np.array([])

    for tau in taus:
        Y_train_pred = np.array([])
        Y_val_pred = np.array([])
        #get the prediction from traing data
        # for test_datum in x_train:
        #     #make a prediction from test data
        #     y = LRLS(test_datum, x_train, y_train, tau)
        #     Y_train_pred = np.append(Y_train_pred,y)
        # #get the predictions from validation data
        # error = Y_train_pred.reshape(Y_train_pred.shape[0],1)-y_train
        # loss_train = np.append(loss_train,np.mean(error**2))
        # print("tau={myTau}, error = {myError}".format(myTau=tau,myError = np.mean(error**2)))
        #calculate val loss
        for val_datum in x_val:
            #make a prediction from val data
            y = LRLS(val_datum, x_val, y_val, tau)
            Y_val_pred = np.append(Y_val_pred,y)
        error_val = Y_val_pred.reshape(Y_val_pred.shape[0],1) - y_val
        loss_val = np.append(loss_val, np.mean(error_val**2))
        print("my method tau={myTau}, error = {myError}".format(myTau=tau, myError=np.mean(error_val ** 2)))

        #verification
        # Prep vectors to return
        train_losses = np.empty_like(taus)
        test_losses = np.empty_like(taus)
        x_test = x_val
        y_test = y_val
        # Compute average loss for each tau
        for (i, t) in enumerate(taus):
            train_predictions = np.array([
                LRLS(datum, x_train, y_train, t)
                for datum in x_train
            ])

            test_predictions = np.array([
                LRLS(datum, x_train, y_train, t)
                for datum in x_test
            ])

            # Error for each datum
            train_errs = (train_predictions - y_train)
            test_errs = (test_predictions - y_test)

            # Use mean squared error
            train_losses[i] = np.mean(train_errs ** 2)
            test_losses[i] = np.mean(test_errs ** 2)

            print(i, t, train_losses[i], test_losses[i])
            print("")
        #verification

    return loss_train, loss_val
    # return train_losses, test_losses
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    #taus = np.logspace(1.0, 3, 200)
    num = 1
    taus = np.logspace(1.0, 1, num)
    # taus = np.array([10])
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    # plt.semilogx(taus,train_losses)
    # # plt.semilogx(test_losses)
    # plt.xlabel("Tau")
    # plt.ylabel("Average Training Loss")
    # plt.show()

    plt.semilogx(taus, test_losses)
    plt.xlabel("Tau")
    plt.ylabel("Average Validation Loss")
    plt.show()

    # print(C)
    # a = np.array([[1,2],[3,4]])
    #
    # print(np.append(a,[[5,6]]))