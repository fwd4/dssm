import pickle
import random
import time

import numpy as np

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', '/tmp/dssm_dump', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.3, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 50000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 1000, "Number of steps in one epoch")

start = time.time()
doc_train_data = pickle.load(open('doc.train.pickle', 'rb')).tocsr()
query_train_data = pickle.load(open('query.train.pickle', 'rb')).tocsr()
doc_test_data = pickle.load(open('doc.test.pickle', 'rb')).tocsr()
query_test_data = pickle.load(open('query.test.pickle', 'rb')).tocsr()
end = time.time()
print("Loading data from HDD to memory: %f s" % (end - start))

TRIGRAM_D = doc_train_data.shape[1]

NEG = 4
BS = 1024

L1_N = 600
L2_N = 300

query_in_shape = np.array([BS, TRIGRAM_D], np.int64)
doc_in_shape = np.array([BS, TRIGRAM_D], np.int64)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


with tf.name_scope('input'):
    # Shape [BS, TRIGRAM_D].
    query_batch = tf.sparse_placeholder(tf.float32, shape=query_in_shape, name='QueryBatch')
    # Shape [BS, TRIGRAM_D]
    doc_batch = tf.sparse_placeholder(tf.float32, shape=doc_in_shape, name='DocBatch')

with tf.name_scope('L1'):
    # Hidden layer 1 [input: cols, output: 300]
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))

    weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    variable_summaries(weight1, 'L1_weights')
    variable_summaries(bias1, 'L1_biases')

    # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
    # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
    doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1) + bias1

    query_l1_out = tf.nn.tanh(query_l1)
    doc_l1_out = tf.nn.tanh(doc_l1)

with tf.name_scope('L2'):
    # Hidden layer 2 [input: 300, output: 300]
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(weight2, 'L2_weights')
    variable_summaries(bias2, 'L2_biases')

    query_l2 = tf.matmul(query_l1_out, weight2) + bias2
    doc_l2 = tf.matmul(doc_l1_out, weight2) + bias2
    query_y = tf.nn.tanh(query_l2)
    doc_y = tf.nn.tanh(doc_l2)

# with tf.name_scope('L3'):
#     # Hidden layer 3 [input: 300, output: 128]
#     l3_par_range = np.sqrt(6.0 / (L2_N + L3_N))
#
#     weight3 = tf.Variable(tf.random_uniform([L2_N, L3_N], -l3_par_range, l3_par_range))
#     bias3 = tf.Variable(tf.random_uniform([L3_N], -l3_par_range, l3_par_range))
#     variable_summaries(weight3, 'L3_weights')
#     variable_summaries(bias3, 'L3_biases')
#
#     query_l3 = tf.matmul(query_l2_out, weight3) + bias3
#     doc_l3 = tf.matmul(doc_l2_out, weight3) + bias3
#     query_y = tf.nn.relu(query_l3)
#     doc_y = tf.nn.relu(doc_l3)

with tf.name_scope('FD_rotate'):
    # Rotate FD+ to produce 50 FD-
    temp = tf.tile(doc_y, [1, 1])

    for i in range(NEG):
        rand = int((random.random() + i) * BS / NEG)
        doc_y = tf.concat(0,
                          [doc_y,
                           tf.slice(temp, [rand, 0], [BS - rand, -1]),
                           tf.slice(temp, [0, 0], [rand, -1])])

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    prod = tf.reduce_sum(tf.mul(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.mul(query_norm, doc_norm)

    cos_sim_raw = tf.truediv(prod, norm_prod)
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    prob = tf.nn.softmax((cos_sim))
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
    tf.scalar_summary('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

merged = tf.merge_all_summaries()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    test_loss_summary = tf.scalar_summary('average_loss', average_loss)
    average_acc = tf.placeholder(tf.float32)
    test_acc_summary = tf.scalar_summary('average_acc', average_acc)


def pull_batch(query_data, doc_data, batch_idx):
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    query_in = query_in.tocoo()
    doc_in = doc_in.tocoo()

    print(query_in.data.shape)
    print(doc_in.data.shape)

    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array(query_in.shape, dtype=np.int64))
    doc_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_in.row, dtype=np.int64), np.array(doc_in.col, dtype=np.int64)]),
        np.array(doc_in.data, dtype=np.float),
        np.array(doc_in.shape, dtype=np.int64))

    return query_in, doc_in


def feed_dict(Train, batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if Train:
        query_in, doc_in = pull_batch(query_train_data, doc_train_data, batch_idx)
    else:
        query_in, doc_in = pull_batch(query_test_data, doc_test_data, batch_idx)
    return {query_batch: query_in, doc_batch: doc_in}


config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test', sess.graph)

    # Actual execution
    for step in range(FLAGS.max_steps):
        batch_idx = step % FLAGS.epoch_steps

        start = time.time()
        loss_v = sess.run(loss, feed_dict=feed_dict(True, batch_idx))
        end = time.time()

        start2 = time.time()
        sess.run(train_step, feed_dict=feed_dict(True, batch_idx))
        end2 = time.time()

        acc_v = sess.run(accuracy, feed_dict=feed_dict(True, batch_idx))
        if batch_idx % 100 == 0:
            summary = sess.run(merged, feed_dict=feed_dict(True, batch_idx))
            train_writer.add_summary(summary, step)
            print("MiniBatch #%-5d | Acc %-3.2f%% | Loss: %-4.3f | FP: %1.4fs | FP+BP: %1.4fs"
                  % (step + 1, acc_v * 100, loss_v, (end - start), end2 - start2))

        if batch_idx == FLAGS.epoch_steps - 1:
            start = time.time()
            epoch_loss = 0
            acc = 0
            for i in range(FLAGS.epoch_steps):
                acc_v, loss_v = sess.run([accuracy, loss], feed_dict=feed_dict(False, i))
                epoch_loss += loss_v
                acc += acc_v

            epoch_loss /= FLAGS.epoch_steps
            acc /= FLAGS.epoch_steps

            loss_summary = sess.run(test_loss_summary, feed_dict={average_loss: epoch_loss})
            acc_summary = sess.run(test_acc_summary, feed_dict={average_acc: acc})
            test_writer.add_summary(loss_summary, step + 1)
            test_writer.add_summary(acc_summary, step + 1)

            end = time.time()
            print ("Epoch     #%-5d | Acc %-3.2f%% | Loss: %-4.3f | Val_Time: %2.4f" %
                   (step / FLAGS.epoch_steps, acc * 100, epoch_loss, end - start))
