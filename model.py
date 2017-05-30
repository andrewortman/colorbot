import argparse
import sys

import tensorflow as tf


def model_fn(features, labels, mode, params, config):
    # TODO: put this in a file and use index_table_from_file instead (set up graph using init op)
    vocabulary = list(" abcdefghijklmnopqrstuvwxyz")

    rnn_cells = params["rnn_cells"]
    rnn_layers = params["rnn_layers"]

    # use the vocabulary lookup table
    vocab = tf.contrib.lookup.index_table_from_tensor(vocabulary)

    # split input strings into characters
    with tf.name_scope("encoder"):
        split = tf.string_split(features, delimiter='')
        # for each character, lookup the index
        encoded = vocab.lookup(split)

        # perform one_hot encoding
        dense_encoding = tf.sparse_tensor_to_dense(encoded, default_value=-1)
        one_hot = tf.one_hot(dense_encoding, len(vocabulary))

        # TODO: better way of computing lengths in the graph?
        lengths = tf.cast(tf.reduce_sum(one_hot, reduction_indices=[1, 2]), tf.int32)

    rnncell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_cells) for _ in range(rnn_layers)])
    rnn_raw_out, _ = tf.nn.dynamic_rnn(cell=rnncell,
                                       inputs=one_hot,
                                       sequence_length=lengths,
                                       dtype=tf.float32)

    batch = tf.range(0, tf.shape(rnn_raw_out)[0])
    indices = tf.stack([batch, lengths - 1], 1)
    rnn_out = tf.gather_nd(rnn_raw_out, indices)

    # output sigmoid
    W = tf.Variable(tf.random_normal(shape=(rnn_cells, 3)), dtype=tf.float32, name="W")
    b = tf.Variable(tf.random_normal(shape=(1, 3)), name="b")

    # predict
    predictions = tf.sigmoid(tf.matmul(rnn_out, W) + b)
    predictions_dict = {
        "colors": predictions,
    }

    # calculate loss
    loss = tf.sqrt(tf.reduce_sum(tf.square(labels-predictions)))/(tf.cast(tf.shape(features)[0], tf.float32))

    # calculate learning rate
    learning_rate_init = params["learning_rate"]
    learning_rate_decay = params["learning_rate_decay"]
    decay_steps = 1e5  # TODO: should this be tunable?
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, decay_steps, learning_rate_decay)

    # training summaries
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar("loss", loss)

    # set up optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)

    # clip gradients (because RNN)
    grad_clip = params["grad_clip"]
    grads_and_vars = [(tf.clip_by_value(g, -grad_clip, grad_clip), v) for g, v in grads_and_vars]
    for g,v in grads_and_vars:
        tf.summary.histogram("grads/"+v.name.replace(":","_"), g)

    # train
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op)


def input_fn(csv_file, batch_size, epochs=None):
    def _input_fn():
        filename_queue = tf.train.string_input_producer([csv_file], num_epochs=epochs)
        reader = tf.TextLineReader(skip_header_lines=True)
        key, value = reader.read(filename_queue)
        record_defaults=[[""],[0.0],[0.0],[0.0]]
        name, red, green, blue = tf.decode_csv(value, record_defaults)
        batches = tf.train.shuffle_batch(
            [name, tf.stack([red/255.0,blue/255.0,green/255.0])],
            batch_size,
            min_after_dequeue=100,
            num_threads=4,
            capacity=1000,
            allow_smaller_final_batch=True)

        return batches[0], batches[1]
    return _input_fn

FLAGS = None
def main(_):
    params = {
        "learning_rate": FLAGS.learning_rate,
        "learning_rate_decay": FLAGS.learning_rate_decay,
        "grad_clip": FLAGS.grad_clip,
        "rnn_layers": FLAGS.rnn_layers,
        "rnn_cells": FLAGS.rnn_cells,
    }
    tf.logging.set_verbosity(tf.logging.INFO)

    network = tf.estimator.Estimator(model_fn=model_fn, params=params)
    while True:
        # train for a few steps
        network.train(input_fn(FLAGS.train_data, FLAGS.batch_size, FLAGS.train_epochs))

        # evaluate
        network.evaluate(input_fn(FLAGS.test_data, FLAGS.batch_size, 1))


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
    parser.add_argument(
        "--train_epochs", type=int, default=10, help="number of epochs to train before evaluations"
    )

    # hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=32, help="minibatch size",
    )
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
