#submission from Peiyao Li 1000674371


import numpy as np

def forward(X_i, w_i, b_i, t, delta):

    new_y =np.dot(w_i, X_i)+b_i
    a = new_y - t
    der_a = np.where(abs(a) <= delta, a, abs(a)/a*delta)
    return new_y, a, der_a

#Overall function that performs the calculation is sln(), assume delta=1
def sln(X, t):

#assume each row of the matrix is a new data, first transpose the matrix so each col is a new data
    X = X.transpose()
    w = np.zeros(X.shape[0])
    b = 0
    delta = 1
    l_rate = 0.01

    cost_last = 0
    dif_cost = 100
    count = 0

    while dif_cost > 0.001 and count < 10000:
        print(count)
        y_i, a, der_a = forward(X, w, b, t, delta)
        cost_i=np.sum(np.where(abs(a) > delta,delta*(abs(a)-0.5*delta),0.5*(a**2)))/X.shape[1]

        dif_cost = abs(cost_last-cost_i)/abs(cost_i)
        print("cost is: {}".format(cost_i))
        print("dif_cost is: {}".format(dif_cost))
        w = w - l_rate * np.dot(der_a, X.transpose())
        b = b - l_rate * np.sum(der_a)/X.shape[1]
        count = 1+count
        cost_last = cost_i

    return w,b

#this section of the code below is a trial initialization of X and y
if __name__ == "__main__":
    p = 20  #number of data points
    q = 5   #number of element in the vector
    X = np.random.randn(p, q)
    w = np.random.randn(q)
    b = 0.6



    target = np.dot(w,X.transpose())+b
    final_w, final_b = sln(X, target)
    print("X is: {}".format(X))
    print("initial w and b are:", w, b)
    print("final w and b are:", final_w, final_b)
    target_new = np.dot(final_w,X.transpose())+final_b
    print("target: {}\ntarget_new: {}".format(target, target_new))









