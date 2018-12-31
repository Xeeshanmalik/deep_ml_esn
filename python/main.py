import matplotlib.pyplot as plt
import numpy as np
import ast
import argparse
import sys
from ESN.ML import ESN


if __name__ == '__main__':

    parser =argparse.ArgumentParser("Multiple Layered Echo State Network")
    parser.add_argument('dataset_file',nargs='?',
                        default=sys.stdin,
                        help='The dataset file to process. Reads from stdin by default.')
    args = parser.parse_args()

    trainLen = 2000
    testLen = 2000
    initLen = 100
    print(args.dataset_file)
    data = []

    with open(args.dataset_file,'r') as f:
        dt = f.readlines()

        for y in dt:
            data.append(ast.literal_eval(y))

    inSize=1
    outSize=1
    resSize=100
    a=0.4

    W_L0=(np.random.rand(resSize,1)-0.5) * 1
    W_reservoir_L0 = np.random.rand(resSize,resSize)-0.5
    W_L1=np.random.rand(resSize,resSize)-0.5
    W_reservoir_L1 = (np.random.rand(resSize,resSize)-0.3)
    W_L2 = np.random.rand(resSize, resSize) - 0.5
    W_reservoir_L2 = (np.random.rand(resSize, resSize) - 0.3)
    W_L3 = np.random.rand(resSize, resSize) - 0.5
    W_reservoir_L3 = (np.random.rand(resSize, resSize) - 0.3)
    W_reservoir_L0 = W_reservoir_L0 * 0.13
    W_reservoir_L1 = W_reservoir_L1 * 0.13
    W_reservoir_L2 = W_reservoir_L2 * 0.13
    W_reservoir_L3 = W_reservoir_L3 * 0.13

    X = np.zeros((resSize, trainLen - initLen))
    Yt = np.transpose(data[initLen+1:trainLen+1])
    x = np.zeros((resSize,1))

    for t in range(1,trainLen):

        u = data[t]

        # Layer 0
        x = np.multiply((1 - a), x) + \
            np.multiply(a, np.tanh(np.add(np.multiply(W_L0, u), np.dot(W_reservoir_L0, x))))

        # Layer 1
        x = ESN(W_L1, W_reservoir_L1, a).esn(x)

        # Layer 2
        x = ESN(W_L2, W_reservoir_L2, a).esn(x)

        # Layer 3
        # x = ESN(W_L3, W_reservoir_L3, a).esn(x)

        # .
        # .
        # .

        # Layer N
        # x = ESN(W_Ln, W_reservoir_Ln, a).esn(x)

        if t > initLen:
            X[:resSize,t-initLen] = np.transpose(x)

    reg = 1e-8
    X_T = np.transpose(X)
    Wout = np.dot(Yt, np.linalg.pinv(X))
    Wout = np.expand_dims(Wout, axis=1)

    Y = np.transpose(np.zeros((outSize, testLen)))
    u = data[trainLen+1]

    for t in range(1,testLen):

        x = np.multiply((1 - a), x) + \
            np.multiply(a, np.tanh(np.add(np.multiply(W_L0, u), np.dot(W_reservoir_L0, x))))

        # Layer 1
        x = ESN(W_L1, W_reservoir_L1, a).esn(x)

        # Layer 2
        x = ESN(W_L2, W_reservoir_L2, a).esn(x)

        # Layer 3
        # x = ESN(W_L3, W_reservoir_L3, a).esn(x)

        # .
        # .
        # .

        # Layer N
        # x = ESN(W_Ln, W_reservoir_Ln, a).esn(x)

        y = np.asscalar(np.dot(np.transpose(Wout), x))

        Y[t] = y
        u = data[trainLen + t + 1]

    plt.title("Target and generated signals y(n) starting at n=0")
    plt.plot(Y)
    plt.plot(data[trainLen + 2:trainLen + testLen + 1])
    plt.legend(['Target signal', 'Free-running predicted signal'], loc='upper left')
    plt.ylim(-0.5, 0.5)
    plt.show()
    errorLen = 100
    mse = np.divide(np.sum(data[trainLen + 2:trainLen + errorLen + 1] - Y[1:errorLen]) ** 2, errorLen)
    print("Mean Squared Error:", (mse))

