import numpy as np

class ESN:
    @staticmethod
    def ESN(data,W,W_reservoir,a):
       return np.multiply(a,data) + \
    np.multiply((1-a),np.tanh(np.add(np.dot(W,data),np.dot(W_reservoir,data))))


