import numpy as np

#####################

#return the graident after summing all the training examples
def dJ_dw(Y,T,X):
    delta=1
    a= Y-T

    n = X.shape[0]

    h_prime = np.copy(a)

    #assign correponding loss values based on the huber loss condition
    h_prime[np.where(abs(a) <= delta)] = a[np.where(abs(a) <= delta)]
    h_prime[np.where(a < -delta)] = -delta
    h_prime[np.where(a > delta)] = delta

    #NOTE: might need to transponse this
    # The final gradient is the sum of each gradient
    return (np.sum(X*h_prime,axis=0)/n).reshape(X.shape[1],1)


# #return the gradient after summing all the training examples
# def y(w,x,b):
#     return np.dot(w.T,x)+b


def dJ_db(Y,T):
    delta = 1
    a = Y - T
    n = Y.shape[0]
    h_prime = np.copy(a)

    # assign corresponding loss values based on the huber loss condition
    h_prime[np.where(abs(a) <= delta)] = a[np.where(abs(a) <= delta)]
    h_prime[np.where(a < -delta)] = -delta
    h_prime[np.where(a > delta)] = delta

    return np.sum(h_prime)/n


#this the predction returned by linear model for all the training example
# note y is n x 1
def pred(w,X,b):
    return np.dot(X,w)+b*np.ones((X.shape[0],1))


def cost_fun(Y,T):
    delta = 1
    a=Y-T
    h = np.copy(a)
    n = Y.shape[0]

    h[np.where(abs(a)<=delta)] = 1/2*a[np.where(abs(a) <= delta)]**2
    h[np.where(abs(a)>delta)] = delta*(abs(a[np.where(abs(a)>delta)]-1/2*delta))

    return np.sum(h)/n


#X is the design matrix T is the true labels
def gradient_descent(X,T):
    #set up some parameters
    t=5000
    alpha = 0.01

    # initialize the weights
    w = np.zeros((X.shape[1],1))
    b = 0
    Y = pred(w, X, b)

    #intialize the parameters
    improvement = 1
    old_cst = 0.1

    #terminate the loop if there is no significant improvement
    while improvement>=0.001:
        Y = pred(w, X, b)
        current_cst = cost_fun(Y,T)

        improvement = abs(current_cst-old_cst)/abs(old_cst)
        old_cst = current_cst

        #update the weights
        w = w - alpha*dJ_dw(Y, T, X)
        b = b - alpha*dJ_db(Y, T)

        print("cost={mycst}".format(mycst=current_cst))

    print("optimal w : {w_s}".format(w_s=w))
    print("optimal b: {b_s}".format(b_s=b))

    return 0


if __name__ == "__main__":
    a=3
    #
    # T = np.array([[0,0,0,1]]).T
    # X = np.array([[0,0],[0,1],[1,0],[1,1]])

    # T = np.array([[1,0]]).T
    # X = np.array([[0],[3]])
    #
    # print(X)
    # print(T)

    p = 20  # number of data points
    q = 10  # number of element in the vector
    X = np.random.randn(p, q)
    T = np.random.randn(p,1)

    print(X)
    print(T)

    gradient_descent(X,T)

    # a=np.array([[1,2,],[3,4],[5,6]])
    #
    # print(a)
    # print(np.sum(a,axis=0)/2)