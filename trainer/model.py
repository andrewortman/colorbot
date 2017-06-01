import tensorflow as tf

def model_fn(features, labels, mode, params):
    input = features["input"]

    # TODO: put this in a file and use index_table_from_file instead (set up graph using init op)
    vocabulary = tf.constant(list(" abcdefghijklmnopqrstuvwxyz"), name="vocab")

    rnn_cells = params["rnn_cells"]
    rnn_layers = params["rnn_layers"]

    # use the vocabulary lookup table
    vocab = tf.contrib.lookup.index_table_from_tensor(vocabulary)

    # split input strings into characters
    with tf.name_scope("encoder"):
        split = tf.string_split(input, delimiter='')
        # for each character, lookup the index
        encoded = vocab.lookup(split)

        # perform one_hot encoding
        dense_encoding = tf.sparse_tensor_to_dense(encoded, default_value=-1)
        one_hot = tf.one_hot(dense_encoding, vocabulary.get_shape()[0])

        # TODO: better way of computing lengths in the graph?
        lengths = tf.cast(tf.reduce_sum(one_hot, reduction_indices=[1, 2]), tf.int32)

    def rnn_layer():
        dropout = params["dropout"]
        l = tf.contrib.rnn.LSTMCell(rnn_cells)
        if dropout > 0.0 and mode is tf.estimator.ModeKeys.TRAIN:
            l = tf.contrib.rnn.DropoutWrapper(l, output_keep_prob=1.0-dropout)
        return l

    rnncell = tf.contrib.rnn.MultiRNNCell([rnn_layer() for _ in range(rnn_layers)])
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
        "color": predictions,
    }

    # export outputs
    exports_dict = {
        "color": tf.estimator.export.PredictOutput(predictions_dict),
    }

    loss = None
    train_op = None

    if mode is not tf.estimator.ModeKeys.PREDICT:
        # calculate loss
        rsme=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(labels, predictions)), axis=1))
        loss = tf.reduce_sum(rsme)/(tf.cast(tf.shape(input)[0], tf.float32))
        tf.summary.histogram("rsme", rsme)
        tf.summary.scalar("loss", loss)

    if mode is tf.estimator.ModeKeys.TRAIN:
        # calculate learning rate
        learning_rate_init = params["learning_rate"]
        learning_rate_decay = params["learning_rate_decay"]
        decay_steps = 1e5  # TODO: should this be tunable?
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, decay_steps, learning_rate_decay)

        # training summaries
        tf.summary.scalar("learning_rate", learning_rate)

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
        export_outputs=exports_dict,
        loss=loss,
        train_op=train_op)

