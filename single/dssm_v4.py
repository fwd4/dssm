import pickle
import random
import time
import sys
import numpy as np
import tensorflow as tf
from datautil import TrainingData
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', './log/20170828/', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
#flags.DEFINE_integer('epoch_steps', 18000, "Number of steps in one epoch.")
#flags.DEFINE_integer('pack_size', 2000, "Number of batches in one pickle pack.")
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")
flags.DEFINE_string('testdata','/data01/dssm/test',"Test Data path")
#flags.DEFINE_string('traindata','/data01/dssm/train',"Training data path")
flags.DEFINE_string('traindata','/data01/dssm/train',"Training data path")
flags.DEFINE_string('modeldir','./model/20170828/',"Model dir")


# load training data for now
start = time.time()
print 'Start to loading test data'
test_data = TrainingData()
test_data.load_data('{}.queryvec'.format(FLAGS.testdata),'{}.docvec'.format(FLAGS.testdata))

print 'Start to loading training data'
train_data = TrainingData()
train_data.load_data('{}.queryvec'.format(FLAGS.traindata),'{}.docvec'.format(FLAGS.traindata))

end = time.time()
print("Loading data from HDD to memory: %.2fs" % (end - start))

TRIGRAM_D = 9289

NEG = 50
BS = 512

L1_N = 400
L2_N = 120

query_in_shape = np.array([BS, TRIGRAM_D], np.int64)
doc_in_shape = np.array([BS, TRIGRAM_D], np.int64)

epoches = train_data.size()/BS
#query_in_shape = np.array([BS, TRIGRAM_D])
#doc_in_shape = np.array([BS, TRIGRAM_D])
print 'query_in_shape ', query_in_shape
print 'doc_in_shape ',doc_in_shape

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar('sttdev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


with tf.name_scope('input'):
    # Shape [BS, TRIGRAM_D].
    #query_batch = tf.sparse_placeholder(tf.float32, shape=query_in_shape, name='QueryBatch')
    query_batch = tf.sparse_placeholder(tf.float32, name='QueryBatch')
    # Shape [BS, TRIGRAM_D]
    #doc_batch = tf.sparse_placeholder(tf.float32, shape=doc_in_shape, name='DocBatch')
    doc_batch = tf.sparse_placeholder(tf.float32, name='DocBatch')

with tf.name_scope('L1'):
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
    
    query_weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    query_bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    variable_summaries(query_weight1, 'L1_query_weights')
    variable_summaries(query_bias1, 'L1_query_biases')

    doc_l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
    doc_weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    doc_bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    
    variable_summaries(doc_weight1, 'L1_doc_weights')
    variable_summaries(doc_bias1, 'L1_doc_biases')

    # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, query_weight1) + query_bias1
    # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
    doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, doc_weight1) + doc_bias1

    query_l1_out = tf.nn.relu(query_l1)
    doc_l1_out = tf.nn.relu(doc_l1)

with tf.name_scope('L2'):
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    query_weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    query_bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(query_weight2, 'L2_query_weights')
    variable_summaries(query_bias2, 'L2_query_biases')

    doc_weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    doc_bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(doc_weight2, 'L2_doc_weights')
    variable_summaries(doc_bias2, 'L2_doc_biases')

    query_l2 = tf.matmul(query_l1_out, query_weight2) + query_bias2
    doc_l2 = tf.matmul(doc_l1_out, doc_weight2) + doc_bias2
    query_y = tf.nn.relu(query_l2,name="query_vec")
    doc_y = tf.nn.relu(doc_l2,name="doc_vec")
    #query_vec = tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True),name="query_vec")
    #doc_vec = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True),name="doc_vec")

with tf.name_scope('FD_rotate'):
    # Rotate FD+ to produce 50 FD-
    temp = tf.tile(doc_y, [1, 1])

    for i in range(NEG):
        rand = int((random.random() + i) * BS / NEG)
        doc_y = tf.concat([doc_y,
                           tf.slice(temp, [rand, 0], [BS - rand, -1]),
                           tf.slice(temp, [0, 0], [rand, -1])],0)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    cos_sim_raw = tf.truediv(prod, norm_prod)
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    prob = tf.nn.softmax((cos_sim))
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
    #tf.scalar_summary('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

# with tf.name_scope('Accuracy'):
#     correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.scalar_summary('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)

'''
def pull_batch(query_data, doc_data, batch_idx):
    # start = time.time()
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    
    if batch_idx == 0:
      print(query_in.getrow(53))
    query_in = query_in.tocoo()
    doc_in = doc_in.tocoo()
    

    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array(query_in.shape, dtype=np.int64))
    doc_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_in.row, dtype=np.int64), np.array(doc_in.col, dtype=np.int64)]),
        np.array(doc_in.data, dtype=np.float),
        np.array(doc_in.shape, dtype=np.int64))

    # end = time.time()
    # print("Pull_batch time: %f" % (end - start))

    return query_in, doc_in


def feed_dict(Train, batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if Train:
        query_in, doc_in = pull_batch(query_train_data, doc_train_data, batch_idx)
    else:
        query_in, doc_in = pull_batch(query_test_data, doc_test_data, batch_idx)
    return {query_batch: query_in, doc_batch: doc_in}
'''

def feed_dict(Train,batch_idx):
    if Train:
        query_in, doc_in = train_data.get_batch(BS,batch_idx)
        if query_in is None or doc_in is None:
            return None
        #print "query_in",query_in
        #print "doc_in",doc_in
        return {query_batch:query_in,doc_batch:doc_in}
    else:
        query_in, doc_in = test_data.get_batch(BS,batch_idx)
        if query_in is None or doc_in is None:
            return None
        return {query_batch:query_in,doc_batch:doc_in}


config = tf.ConfigProto(intra_op_parallelism_threads=32,inter_op_parallelism_threads=16)  # log_device_placement=True)
config.gpu_options.allow_growth = True
#if not FLAGS.gpu:
#config = tf.ConfigProto(device_count= {'GPU' : 0})

with tf.Session(config=config) as sess, tf.device('/cpu:0'):
    sess.run(tf.initialize_all_variables())
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test', sess.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    # Actual execution
    start = time.time()
    # fp_time = 0
    # fbp_time = 0
    for step in range(FLAGS.max_steps):
        batch_idx = step % epoches

        if batch_idx % 100 == 0:
            progress = 100.0 * batch_idx / epoches
            sys.stdout.write("\r%.2f%% Epoch" % progress)
            sys.stdout.flush()

        # t1 = time.time()
        # sess.run(loss, feed_dict = feed_dict(True, batch_idx))
        # t2 = time.time()
        # fp_time += t2 - t1
        # #print(t2-t1)
        # t1 = time.time()
        fd = feed_dict(True, batch_idx)
        if fd is None:
            continue
        sess.run(train_step, feed_dict=fd)
        # t2 = time.time()
        # fbp_time += t2 - t1
        # #print(t2 - t1)
        # if batch_idx % 2000 == 1999:
        #     print ("MiniBatch: Average FP Time %f, Average FP+BP Time %f" %
        #        (fp_time / step, fbp_time / step))


        if batch_idx == epoches - 1:
            end = time.time()
            epoch_loss = 0
            for i in range(epoches):
                loss_v = sess.run(loss, feed_dict=feed_dict(True, i))
                epoch_loss += loss_v

            epoch_loss /= epoches
            train_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(train_loss, step + 1)

            # print ("MiniBatch: Average FP Time %f, Average FP+BP Time %f" %
            #        (fp_time / step, fbp_time / step))
            #
            print ("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                    (step / epoches, epoch_loss, end - start))

            epoch_loss = 0
            for i in range(int(test_data.size()/BS)):
                loss_v = sess.run(loss, feed_dict=feed_dict(False, i))
                epoch_loss += loss_v

            epoch_loss /= float(int(test_data.size()/BS))

            test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            test_writer.add_summary(test_loss, step + 1)
            saver.save(sess,FLAGS.modeldir,global_step=step)
            start = time.time()
            print ("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
                   (step / epoches, epoch_loss, start - end))

