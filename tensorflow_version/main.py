import requests
import numpy as np
import tensorflow as tf
import ast
from ml_esn_cells import MLESNCell
import argparse
import sys


def MackeyGlass(tr_size=500, washout_size=50, units=30, connectivity=0.2, scale=0.7, elements=2000):
  print("Fetching data...")
  data_str = []
  with open(args.dataset_file, 'r') as f:
    dt = f.readlines()
    for y in dt:
      data_str.append(ast.literal_eval(y))

  data = map(float, data_str[:elements])
  data_t = tf.reshape(tf.constant(data), [1, elements, 1])
  esn = MLESNCell(units, connectivity, scale)

  print("Building graph...")
  outputs, final_state = tf.nn.dynamic_rnn(esn, data_t, dtype=tf.float32)
  washed = tf.squeeze(tf.slice(outputs, [0, washout_size, 0], [-1, -1, -1]))

  with tf.Session() as S:
    S.run(tf.global_variables_initializer())

    print("Computing embeddings...")
    res = S.run(washed)

    print("Computing direct solution...")
    state = np.array(res)
    tr_state = np.mat(state[:tr_size])
    ts_state = np.mat(state[tr_size:])
    wout = np.transpose(np.mat(data[washout_size+1:tr_size+washout_size+1]) * np.transpose(np.linalg.pinv(tr_state)))

    print("Testing performance...")
    ts_out = np.mat((np.transpose(ts_state * wout).tolist())[0][:-1])
    ts_y = np.mat(data[washout_size+tr_size+1:])

    ts_mse = np.mean(np.square(ts_y - ts_out))

  print("Test MSE: " + str(ts_mse))

if __name__ == "__main__":

  parser = argparse.ArgumentParser("Multiple Layered Echo State Network")
  parser.add_argument('dataset_file', nargs='?',
                      default=sys.stdin,
                      help='The dataset file to process. Reads from stdin by default.')
  args = parser.parse_args()

  MackeyGlass()