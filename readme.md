# Deep Multiple Layered Echo State Network

A novel architecture and learning algorithm for a multilayered echo state network (ML-ESM). The addition
of multiple layers of reservoir are shown to provide a more robust alternative to conventional RC networks.

Malik, Z. K., Hussain, A., & Wu, Q. J. (2017). Multilayered echo state machine: a novel architecture and algorithm. IEEE Transactions on cybernetics, 47(4), 946-959.


How to use this algorithm.


ADD ESN package into your project.


## Layer 1
x = E.ESN(x,W_L1, W_reservoir_L1, a)

## Layer 2

x = E.ESN(x,W_L2, W_reservoir_L2, a)


pip project: https://packaging.python.org/tutorials/packaging-projects/