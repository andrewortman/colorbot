import tensorflow as tf
import argparse
import sys

import model
import util

def input_fn(csv_file, batch_size, epochs=None):
    def _input_fn():
        filename_queue = tf.train.string_input_producer([csv_file], num_epochs=epochs)
        reader = tf.TextLineReader(skip_header_lines=True)
        _, value = reader.read(filename_queue)
        record_defaults = [[""], [0.0], [0.0], [0.0]]
        name, red, green, blue = tf.decode_csv(value, record_defaults)
        batches = tf.train.shuffle_batch(
            [name, tf.stack([red/255.0, green/255.0, blue/255.0])],
            batch_size,
            min_after_dequeue=100,
            num_threads=4,
            capacity=1000,
            allow_smaller_final_batch=True)

        return {"input": batches[0]}, batches[1]

    return _input_fn


FLAGS = None

def main(_):
    params = {
        "learning_rate": FLAGS.learning_rate,
        "grad_clip": FLAGS.grad_clip,

        "rnn_cells": FLAGS.rnn_cells,
        "rnn_dropout": FLAGS.rnn_dropout,
        "output_cells": FLAGS.output_cells,
        "output_dropout": FLAGS.output_dropout,
    }
    tf.logging.set_verbosity(tf.logging.INFO)

    network = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=FLAGS.job_dir, params=params)
    while True:
        # train for a few steps
        network.train(input_fn(FLAGS.train_data, FLAGS.batch_size, FLAGS.train_epochs))

        # evaluate
        network.evaluate(input_fn(FLAGS.test_data, FLAGS.batch_size, 1))

        export_dir = None
        if FLAGS.export_dir is not None:
            export_dir = FLAGS.export_dir
        elif FLAGS.job_dir is not None:
            export_dir = FLAGS.job_dir + "/exports"

        if export_dir is None:
            print "Skipping model export (set job-dir or export-dir)"
        else:
            print "Exporting model..."
            feature_spec = {"input": tf.constant("", shape=[1], dtype=tf.string)}
            serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
            export_dir = network.export_savedmodel(export_dir, serving_input_fn)
            print "Exported to " + export_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train-data", type=str, default="./data/train.csv", help="Path to the training data.")
    parser.add_argument(
        "--test-data", type=str, default="./data/test.csv", help="Path to the test data.")
    parser.add_argument(
        "--job-dir", type=str, default=None, help="Path to store the model and checkpoints during training",
    )
    parser.add_argument(
        "--export-dir", type=str, default=None, help="Path to store exported models (defaults to job-dir + /exports)",
    )
    parser.add_argument(
        "--train-epochs", type=int, default=10, help="number of epochs to train before evaluations and exports"
    )

    # hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="minibatch size",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-3,
        help="learning rate for ADAM optimizer",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=0.1,
        help="gradient clip (absolute)",
    )

    parser.add_argument(
        "--rnn-cells", type=util.layer_list_type(allow_empty=False), default="32,32",
        help="rnn cell layer sizes seperated with commas",
    )
    parser.add_argument(
        "--rnn-dropout", type=float, default=0.5,
        help="dropout (0->1) to add to rnn outputs. 0 = no dropout, 1.0 = full dropout",
    )

    parser.add_argument(
        "--output-cells", type=util.layer_list_type(allow_empty=True), default="",
        help="additional output layer sizes seperated with commas",
    )
    parser.add_argument(
        "--output-dropout", type=float, default=0.5,
        help="dropout (0->1) to add to non-final outputs. 0 = no dropout, 1.0=full_dropout",
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
