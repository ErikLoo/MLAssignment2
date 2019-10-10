import numpy as np

#####################

#return the graident with respect to w, after summing all the training examples
def dJ_dw(Y,T,X):
    delta=1
    a= Y-T

    n = X.shape[0]

    h_prime = np.copy(a)

    #assign correponding value based on the huber loss condition
    h_prime[np.where(abs(a) <= delta)] = a[np.where(abs(a) <= delta)]
    h_prime[np.where(a < -delta)] = -delta
    h_prime[np.where(a > delta)] = delta

    # The final gradient is the sum of each gradient /n
    return (np.sum(X*h_prime, axis=0)/n).reshape(X.shape[1], 1)

#return the graident with respect to b, summing over all the traing examples
def dJ_db(Y,T):
    delta = 1
    a = Y - T
    n = Y.shape[0]
    h_prime = np.copy(a)

    # assign corresponding value based on the huber loss condition
    h_prime[np.where(abs(a) <= delta)] = a[np.where(abs(a) <= delta)]
    h_prime[np.where(a < -delta)] = -delta
    h_prime[np.where(a > delta)] = delta

    #the final gradient is the sum of each gradient /n
    return np.sum(h_prime)/n


#this the predction returned by linear model for all the training example
# note y is n x 1
def pred(w,X,b):
    return np.dot(X,w)+b*np.ones((X.shape[0],1))

#this is the cost function
def cost_fun(Y,T):
    delta = 1
    a=Y-T
    h = np.copy(a)
    n = Y.shape[0]

    h[np.where(abs(a)<=delta)] = 1/2*a[np.where(abs(a) <= delta)]**2
    h[np.where(abs(a)>delta)] = delta*(abs(a[np.where(abs(a)>delta)]-1/2*delta))

    return np.sum(h)/n


#X is the design matrix T is the true label
def gradient_descent(X,T):
    #set up some parameters
    #learning rate
    alpha = 0.01

    # initialize the weights
    w = np.zeros((X.shape[1],1))
    b = 0

    #intialize the parameters
    improvement = 1
    old_cst = 0.1

    #terminate the loop if there is no significant improvement, improvement lesws than 0.01
    while improvement>=0.001:
        Y = pred(w, X, b)
        current_cst = cost_fun(Y,T)

        #calculating the differential cost
        improvement = abs(current_cst-old_cst)/abs(old_cst)
        old_cst = current_cst

        #update the weights
        w = w - alpha*dJ_dw(Y, T, X)
        b = b - alpha*dJ_db(Y, T)

        #print out
        print("cost={mycst}".format(mycst=current_cst))

    print("optimal w : {w_s}".format(w_s=w.T))
    print("optimal b: {b_s}".format(b_s=b))

    return 0


if __name__ == "__main__":

    M = 5  # number of data points
    N = 5  # number of features
    X = np.random.randn(M, N)
    T = np.random.randn(M,1)

    # X = np.array([[0],[1]])
    # T = np.array([[1],[0]])

    #print out X and T for clarity
    print(X)
    print(T)

    gradient_descent(X, T)
