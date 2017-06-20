import tensorflow as tf

def model_fn(features, labels, mode, params):
    input = features["input"]

    # TODO: put this in a file and use index_table_from_file instead (set up graph using init op)
    vocabulary = tf.constant(list(" abcdefghijklmnopqrstuvwxyz"), name="vocab")

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

        # TODO: better way of computing sequence lengths in the graph?
        lengths = tf.cast(tf.reduce_sum(one_hot, reduction_indices=[1, 2]), tf.int32)

    def rnn_layer(size):
        keep_prob = 1.0 - params["rnn_dropout"]
        l = tf.contrib.rnn.GRUCell(size)
        if keep_prob < 1.0 and mode is tf.estimator.ModeKeys.TRAIN:
            l = tf.contrib.rnn.DropoutWrapper(l, output_keep_prob=keep_prob)
        return l

    rnn_layers = []

    rnn_cell_sizes = params["rnn_cells"]
    for size in rnn_cell_sizes:
        rnn_layers.append(rnn_layer(size))

    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
    rnn_raw_out, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=one_hot,
                                       sequence_length=lengths,
                                       dtype=tf.float32,
                                       scope="rnn_layers")

    with tf.name_scope("rnn_output_relevant"):
        # get the last relevant output from the rnn outputs
        batch = tf.range(0, tf.shape(rnn_raw_out)[0]) # generate 0->batch_size
        coordinates = tf.stack([batch, lengths - 1], 1) # stack the 0->batch_size sequence with the sequence lengths
        rnn_out = tf.gather_nd(rnn_raw_out, coordinates) # perform a gather using those coordinates

    # output sigmoid layers
    output_cell_sizes = params["output_cells"]
    output_dropout = params["output_dropout"]

    def output_layer(last_layer, last_layer_size, layer_size, dropout):
        W = tf.Variable(tf.random_uniform((last_layer_size, layer_size), -1, 1), dtype=tf.float32, name="W")
        b = tf.Variable(tf.random_uniform((1, layer_size), -1, 1), name="b")
        sig = tf.sigmoid(tf.matmul(last_layer, W) + b)
        output = sig
        keep_prob = 1.0-dropout
        if keep_prob > 0.0:
            output = tf.nn.dropout(output, 1.0-dropout)

        return output

    last_layer = rnn_out
    last_layer_size = rnn_cell_sizes[-1]
    with tf.name_scope("output_layers"):
        for idx in range(0, len(output_cell_sizes)):
            with tf.name_scope("layer"+str(idx)):
                last_layer = output_layer(last_layer, last_layer_size, output_cell_sizes[idx], output_dropout)
                last_layer_size = output_cell_sizes[idx]

        # final prediction output
        with tf.name_scope("final"):
            predictions = output_layer(last_layer, last_layer_size, 3, False)


    # predict
    predictions_dict = {
        "color": predictions,
    }

    # export outputs
    exports_dict = {
        "color": tf.estimator.export.PredictOutput(predictions_dict),
    }

    loss = None
    train_op = None

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        # calculate loss
        rsme=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(labels, predictions)), axis=1))
        loss = tf.reduce_sum(rsme)/(tf.cast(tf.shape(input)[0], tf.float32))

        # metrics for each mode (train, eval)
        tf.summary.scalar("loss/"+mode, loss)
        tf.summary.histogram("loss/"+mode, rsme)

    if mode is tf.estimator.ModeKeys.TRAIN:
        learning_rate = params["learning_rate"]

        # set up optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)

        # clip gradients to help with exploding gradients in RNN's
        grad_clip = params["grad_clip"]
        grads_and_vars = [(tf.clip_by_value(g, -grad_clip, grad_clip), v) for g, v in grads_and_vars]

        # train
        global_step = tf.train.get_global_step()
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # training summaries (picked up by tf.Estimator)
        tf.summary.scalar("learning_rate", learning_rate)
        for g,v in grads_and_vars:
            tf.summary.histogram("grads/"+v.name.replace(":","_"), g)
        for v in tf.trainable_variables():
            tf.summary.histogram("vars/"+v.name.replace(":","_"), v)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        export_outputs=exports_dict,
        loss=loss,
        train_op=train_op,
    )

