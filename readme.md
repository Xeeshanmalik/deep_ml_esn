# Deep Multiple Layered Echo State Network

A novel architecture and learning algorithm for a multilayered echo state network (ML-ESM). The addition
of multiple layers of reservoir are shown to provide a more robust alternative to conventional RC networks.

````
Malik, Z. K., Hussain, A., & Wu, Q. J. (2017). Multilayered echo state machine: 
a novel architecture and algorithm. IEEE Transactions on cybernetics, 47(4), 946-959.


````
This project consist of deep multilayered ESN implementation having

1) Matlab Version
2) Python Version
3) Tensorflow Version

## To run both tensorflow and python version

````
chmod +x ./run_train_test.sh
./run_train_test.sh

````
## How to currently use this algorithm in your own project.

````
pip install ESN
````
## ADD ESN package into your project.

````
from ESN.ML import ESN
````

## Layer 0

### Standard State of The Art ESN
````
        x = np.multiply((1 - a), x) + \
            np.multiply(a, np.tanh(np.add(np.multiply(W_L0, u), np.dot(W_reservoir_L0, x))))
````

## Layer 1

L1 = ESN(W_L1, W_reservoir_L1, a)(x)

## Layer 2

L2 = ESN(W_L2, W_reservoir_L2, a)(L1)

. <br />
. <br />
. <br />
. <br />

## Layer N

Ln = ESN(W_Ln, W_reservoir_Ln, a)(Ln-1)

Output of The Project on McKayGlass Timeseries Dataset

![alt text](https://github.com/Xeeshanmalik/deep_ml_esn/blob/master/data/single.png)


## Limitations

1) The weights everytime are initialized randomly but on scale recommendation is to optimize, tune and fix the initialization of both
   internal, external and reservoirs weights like any other networks excepts the readouts.
