import argparse
import sys

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

def model_fn(features, labels, mode, params):
    # TODO: put this in a file and use index_table_from_file instead (set up graph using init op)
    vocabulary = list(" abcdefghijklmnopqrstuvwxyz")

    rnn_cells = params["rnn_cells"]
    rnn_layers = params["rnn_layers"]

    with tf.name_scope("encoder"):
        # use the vocabulary lookup table
        vocab = tf.contrib.lookup.index_table_from_tensor(vocabulary)

        # split input strings into characters
        split = tf.string_split(features, delimiter='')
        # for each character, lookup the index
        encoded = vocab.lookup(split)

        # perform one_hot encoding
        dense_encoding = tf.sparse_tensor_to_dense(encoded, default_value=-1)
        one_hot = tf.one_hot(dense_encoding, len(vocabulary))

        # TODO: better way of computing lengths in the graph?
        lengths = tf.cast(tf.reduce_sum(one_hot, reduction_indices=[1, 2]), tf.int32)

    with tf.name_scope("recurrent"):
        rnncell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_cells) for i in range(rnn_layers)])
        rnn_raw_out, _ = tf.nn.dynamic_rnn(cell=rnncell,
                                                 inputs=one_hot,
                                                 sequence_length=lengths,
                                                 dtype=tf.float32)

        with tf.name_scope("last_relevant"):
            batch = tf.range(0, tf.shape(rnn_raw_out)[0])
            indices = tf.stack([batch, lengths - 1], 1)
            rnn_out = tf.gather_nd(rnn_raw_out, indices)

    with tf.name_scope("predictions"):
        W = tf.Variable(tf.random_normal(shape=(rnn_cells, 3)), dtype=tf.float32, name="W")
        b = tf.Variable(tf.random_normal(shape=(1, 3)), "b")

        predictions = tf.sigmoid(tf.matmul(rnn_out, W) + b)
        predictions_dict = {
            "colors": predictions,
        }

    with tf.name_scope("loss"):
        rmse = tf.metrics.root_mean_squared_error(labels, predictions)
        losses = {
            "rmse": rmse,
        }

    with tf.name_scope("train"):
        learning_rate_init = params["learning_rate"]
        learning_rate_decay = params["learning_rate_decay"]
        decay_steps = 1e5 # TODO: should this be tunable?
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, decay_steps, learning_rate_decay)

        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(rmse)

        grad_clip = params["grad_clip"]
        grads_and_vars = [(tf.clip_by_value(g, -grad_clip, grad_clip),v) for g,v in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars)

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=rmse,
        train_op=train_op,
        eval_metric_ops=losses)

FLAGS = None
def main(_):
    params = {
        "learning_rate": FLAGS.learning_rate,
        "learning_rate_decay": FLAGS.learning_rate_decay,
        "grad_clip": FLAGS.grad_clip,
        "rnn_layers": FLAGS.rnn_layers,
        "rnn_cells": FLAGS.rnn_cells,
    }

    print params
    print FLAGS.output
    nn = tf.estimator.Estimator(model_fn, params=params)
    print "hello world"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
      "--train_data", type=str, default="./data/train.csv", help="Path to the training data.")
    parser.add_argument(
      "--test_data", type=str, default="./data/test.csv", help="Path to the test data.")
    parser.add_argument(
      "--output", type=str, default="./models", help="Path to store the model and checkpoints during training",
    )

    # hyperparameters
    parser.add_argument(
      "--learning_rate", type=float, default=1e-3, help="initial learning rate",
    )
    parser.add_argument(
      "--learning_rate_decay", type=float, default=0.999, help="initial learning rate decay over time",
    )
    parser.add_argument(
      "--grad_clip", type=float, default=1.0, help="gradient clip (absolute)",
    )
    parser.add_argument(
      "--rnn_layers", type=int, default=3, help="number of rnn layers",
    )
    parser.add_argument(
      "--rnn_cells", type=int, default=128, help="number of rnn cells per layer",
    )
    # todo: rnn cell type

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
