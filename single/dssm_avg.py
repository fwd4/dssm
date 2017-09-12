import random
import time
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datautil import TrainingData
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', './log/20170828_related/', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_bool('gpu', 0, "Enable GPU or not")
flags.DEFINE_string('testdata','/data01/dssm/test',"Test Data path")
flags.DEFINE_string('traindata','/data01/dssm/train',"Training data path")
flags.DEFINE_string('modeldir','./model/20170828_related/',"Model dir")
flags.DEFINE_integer('wordhashdim',-1,'wordhash dimension')
flags.DEFINE_integer('querynum',10,'how many queries is used to match a document')
# load training data for now
start = time.time()
print 'Start to loading test data ',FLAGS.testdata
test_data = TrainingData()
test_data.load_data('{}.queryvec'.format(FLAGS.testdata),'{}.docvec'.format(FLAGS.testdata))

print 'Start to loading training data ',FLAGS.traindata
train_data = TrainingData()
train_data.load_data('{}.queryvec'.format(FLAGS.traindata),'{}.docvec'.format(FLAGS.traindata))

end = time.time()
print("Loading data from HDD to memory: %.2fs" % (end - start))

TRIGRAM_D = FLAGS.wordhashdim

NEG = 50
BS = 512

L1_N = 256
L2_N = 128

query_in_shape = np.array([BS, TRIGRAM_D], np.int64)
doc_in_shape = np.array([BS, TRIGRAM_D], np.int64)

epoches = train_data.size()/BS
batch_num = train_data.size()/BS
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
    query_batch = tf.sparse_placeholder(tf.float32, name='QueryBatch')
    # Shape [BS, TRIGRAM_D]
    doc_batch = tf.sparse_placeholder(tf.float32, name='DocBatch')

with tf.name_scope('L1'):
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
    weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    variable_summaries(weight1, 'L1_weights')
    variable_summaries(bias1, 'L1_biases')

    # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
    # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
    doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1) + bias1

    query_l1_out = tf.nn.relu(query_l1)
    doc_l1_out = tf.nn.relu(doc_l1)

with tf.name_scope('L2'):
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(weight2, 'L2_weights')
    variable_summaries(bias2, 'L2_biases')

    query_l2 = tf.matmul(query_l1_out, weight2) + bias2
    doc_l2 = tf.matmul(doc_l1_out, weight2) + bias2
    query_y = tf.nn.tanh(query_l2,name="query_vec")
    doc_y = tf.nn.tanh(doc_l2,name="doc_vec")

with tf.name_scope('FD_rotate'):
    # Rotate FD+ to produce 50 FD-
    temp = tf.tile(doc_y, [1, 1]) # temp equals doc_y here. just a replica

    for i in range(NEG):
        rand = int((random.random() + i) * BS / NEG)
        doc_y = tf.concat([doc_y,
                           tf.slice(temp, [rand, 0], [BS - rand, -1]),
                           tf.slice(temp, [0, 0], [rand, -1])],0)
        #each time the temp(replicated from doc_y) tensor will be splitted into 2 parts randomly(the split point is randomly choosen) and concated inversely
        #Simple Example: [1,2,3,4,5] => [4,5] [1,2,3]=>[4,5,1,2,3]
    #At the end loop doc_y becomes a (NEG+1)*BatchSize rows of vectors
with tf.name_scope('AvgQueryVec'):
    query_y = tf.reshape(query_y,[BS,FLAGS.querynum,L2_N])
    query_y = tf.reduce_mean(query_y,axis=1)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    cos_sim_raw = tf.truediv(prod, norm_prod) #cos_sim_raw is a (NEG+1)*BS  row vector
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * 20 # transform to a BS x (NEG+1)  matrix and then compute the softmax on BS vectors, each vector has NEG+1 elements. The first element is the prob (cosine) of ground truth while others are negative samples. 

with tf.name_scope('Loss'):
    # Train Loss
    prob = tf.nn.softmax((cos_sim))
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
    #tf.scalar_summary('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    average_acc = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)
    acc_summary = tf.summary.scalar('average_acc', average_acc)

def feed_dict(Train,batch_idx):
    if Train:
        query_in = train_data.get_query_batch(BS*FLAGS.querynum,batch_idx,FLAGS.wordhashdim)
        doc_in = train_data.get_doc_batch(BS,batch_idx,FLAGS.wordhashdim)
        if query_in is None or doc_in is None:
            return None
        return {query_batch:query_in,doc_batch:doc_in}
    else:
        query_in = test_data.get_query_batch(BS*FLAGS.querynum,batch_idx,FLAGS.wordhashdim)
        doc_in = test_data.get_doc_batch(BS,batch_idx,FLAGS.wordhashdim)
        if query_in is None or doc_in is None:
            return None
        return {query_batch:query_in,doc_batch:doc_in}


config = tf.ConfigProto()  # log_device_placement=True)
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
        #Train
        for batch_idx in tqdm(range(batch_num)):
            fd = feed_dict(True, batch_idx)
            if fd is None:
                continue
            sess.run(train_step, feed_dict=fd)
        #Evaluation on training set
        epoch_loss = 0
        epoch_acc = 0
        for batch_idx in range(batch_num):
            loss_v = sess.run(loss, feed_dict=feed_dict(True,batch_idx))
            epoch_loss += loss_v
            #epoch_acc += acc_v
            #acc_v,loss_v = sess.run(accuracy,loss, feed_dict=feed_dict(True,batch_idx))
            #epoch_loss += loss_v
            #epoch_acc += acc_v

        epoch_loss /= batch_num
        epoch_acc /= batch_num
        train_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss,average_acc:epoch_acc})
        train_writer.add_summary(train_loss, step + 1)
        #train_writer.add_summary(train_acc, step + 1)
        #train_acc,train_loss = sess.run(acc_summary,loss_summary, feed_dict={average_loss: epoch_loss,average_acc:epoch_acc})
        #train_writer.add_summary(train_loss, step + 1)
        #train_writer.add_summary(train_acc, step + 1)
        train_loss = epoch_loss
        epoch_loss = 0
        epoch_acc = 0
        for batch_idx in range(int(test_data.size()/BS)):
            loss_v = sess.run(loss, feed_dict=feed_dict(False, batch_idx))
            epoch_loss += loss_v
            #epoch_acc += acc_v
        epoch_loss /= float(int(test_data.size()/BS))
        epoch_acc /= float(int(test_data.size()/BS))
        test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss,average_acc:epoch_acc})
        test_writer.add_summary(test_loss, step + 1)
        #test_writer.add_summary(test_cc, step + 1)
        saver.save(sess,FLAGS.modeldir,global_step=step)
        test_loss = epoch_loss
        train_acc = 0
        test_acc = 0
        #train_loss = 0
        #test_loss = 0
        print ("Step #%d:\n Train Loss: %4.3f, ACC: %4.3f | Test  Loss: %4.3f ACC:%4.3f " %
               (step,train_loss,train_acc,test_loss,test_acc))

