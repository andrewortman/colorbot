import tensorflow as tf

vocab_name = "./data/vocab.txt"
train_name = "./data/train.csv"
test_name = "./data/test.csv"

def encoder(instr):
    vocab = tf.contrib.lookup.index_table_from_file(vocab_name)
    split = tf.string_split(instr, delimiter='')
    indices = vocab.lookup(split)
    dense_indices = tf.sparse_tensor_to_dense(indices, default_value=-1)
    one_hot = tf.one_hot(dense_indices, tf.cast(vocab.size(), tf.int32))
    lengths = tf.reduce_sum(one_hot, reduction_indices=[1,2])
    return (one_hot, lengths)


with tf.Session() as sess:
    instr = tf.placeholder(tf.string)
    with tf.name_scope('encoder'):
        encoded_input = encoder(instr)

    sess.run(tf.tables_initializer())
    encode = encoded_input
    print sess.run(encode, {instr: ["hello world", "hello"]})
