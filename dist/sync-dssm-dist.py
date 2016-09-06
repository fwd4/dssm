import pickle
import random
import time

import numpy as np

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags for defining the tf.train.ClusterSpec
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_integer("num_workers", "", "Number of workers ")

flags.DEFINE_string('summaries_dir', '/tmp/dssm-dist', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 50000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 1000, "Number of steps in one epoch")
# FLAGS.max_steps = FLAGS.num_workers
# FLAGS.epoch_steps = FLAGS.num_workers
# flags.DEFINE_integer('gama', 20, 'Multiply cosine similarity by gama before softmax')
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")

start = time.time()
doc_train_data = pickle.load(open('doc.train.pickle', 'rb')).tocsr()
query_train_data = pickle.load(open('query.train.pickle', 'rb')).tocsr()
doc_test_data = pickle.load(open('doc.test.pickle', 'rb')).tocsr()
query_test_data = pickle.load(open('query.test.pickle', 'rb')).tocsr()
end = time.time()
print("Number of batches per epoch per worker: %d " % FLAGS.epoch_steps)
print("Loading data from HDD to memory: %f s" % (end - start))

FLAGS.learning_rate /= np.sqrt(FLAGS.num_workers)
FLAGS.max_steps //= FLAGS.num_workers
FLAGS.epoch_steps //= FLAGS.num_workers

BS = 1024 # // FLAGS.num_workers

TRIGRAM_D = doc_train_data.shape[1]
NEG = 50

L1_N = 400
L2_N = 120
GAMA = 20


def pull_batch(query_data, doc_data, batch_idx):
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    cols = np.unique(np.concatenate((query_in.tocoo().col.T, doc_in.tocoo().col.T), axis=0))
    # print(query_in.shape)
    # print(doc_in.shape)
    query_in = query_in[:, cols].tocoo()
    doc_in = doc_in[:, cols].tocoo()

    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array(query_in.shape, dtype=np.int64))
    doc_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_in.row, dtype=np.int64), np.array(doc_in.col, dtype=np.int64)]),
        np.array(doc_in.data, dtype=np.float),
        np.array(doc_in.shape, dtype=np.int64))

    return query_in, doc_in, cols


def feed_dict(Train, batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if Train:
        query_in, doc_in, cols = pull_batch(query_train_data, doc_train_data, batch_idx)
    else:
        query_in, doc_in, cols = pull_batch(query_test_data, doc_test_data, batch_idx)
    return {query_batch: query_in, doc_batch: doc_in, dense_cols: cols}


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


query_in_shape = np.array([BS, TRIGRAM_D]).astype('int64')
doc_in_shape = np.array([BS, TRIGRAM_D]).astype('int64')

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")

# Create a cluster from the parameter server and worker hosts
cluster = tf.train.ClusterSpec({"worker": worker_hosts, "ps": ps_hosts})
# Cteate and start a server for the local task.
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(  # cluster = cluster)):
            worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # quit()
        with tf.name_scope('input'):
            # Shape [BS, TRIGRAM_D].
            query_batch = tf.sparse_placeholder(tf.float32, shape=query_in_shape, name='QueryBatch')
            # Shape [BS, TRIGRAM_D]
            doc_batch = tf.sparse_placeholder(tf.float32, shape=doc_in_shape, name='DocBatch')

            dense_cols = tf.placeholder(tf.int64, shape=None, name='ActualCols')

        with tf.name_scope('L1'):
            # Hidden layer 1 [input: cols, output: 300]
            l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
            
            #partitioner = tf.variable_axis_size_partitioner(16<<20-1)
            #weight1 = tf.get_variable("weight", [TRIGRAM_D, L1_N], 
            #                          initializer=tf.random_uniform_initializer(
            #                                     minval=-l1_par_range,
            #                                     maxval=l1_par_range,
            #                                      dtype=tf.float32),
            #                          partitioner=partitioner)
            weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range),
                                  name='weight')
            bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range),
                                name='bias')
            variable_summaries(weight1, 'L1_weights')
            variable_summaries(bias1, 'L1_biases')
            with tf.device(weight1.device):
                dense_w1 = tf.gather(weight1, dense_cols, name="L1_dense_w")
            # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
            query_l1 = tf.sparse_tensor_dense_matmul(query_batch, dense_w1) + bias1
            # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
            doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, dense_w1) + bias1

            query_l1_out = tf.nn.relu(query_l1)
            doc_l1_out = tf.nn.relu(doc_l1)

        with tf.name_scope('L2'):
            # Hidden layer 2 [input: 300, output: 300]
            l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
            weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range),
                                  name='weight')
            bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range),
                                name='bias')
            variable_summaries(weight2, 'L2_weights')
            variable_summaries(bias2, 'L2_biases')

            query_l2 = tf.matmul(query_l1_out, weight2) + bias2
            doc_l2 = tf.matmul(doc_l1_out, weight2) + bias2
            query_y = tf.nn.relu(query_l2)
            doc_y = tf.nn.relu(doc_l2)

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
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * GAMA

        with tf.name_scope('Loss'):
            # Train Loss
            prob = tf.nn.softmax((cos_sim))
            hit_prob = tf.slice(prob, [0, 0], [-1, 1])
            loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
            tf.scalar_summary('loss', loss)

        with tf.name_scope('Training'):
            # Optimizer
            # train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)\
            #    .minimize(loss)
            opt = tf.train.AdagradOptimizer(FLAGS.learning_rate)
            opt = tf.train.SyncReplicasOptimizer(opt,
                                                 replicas_to_aggregate=FLAGS.num_workers,
                                                 total_num_replicas=FLAGS.num_workers,
                                                 replica_id=FLAGS.task_index,
                                                 name="dssm_sync_replicas")
            train_step = opt.minimize(loss, global_step=global_step)

        # merged = tf.merge_all_summaries()

        with tf.name_scope('Test'):
            average_loss = tf.placeholder(tf.float32)
            loss_summary = tf.scalar_summary('average_loss', average_loss)

        if (FLAGS.task_index == 0):
            chief_queue_runner = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op()

        init_op = tf.initialize_all_variables()

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/dssm-dist",
                             init_op=init_op,
                             saver=None,
                             global_step=global_step,
                             summary_op=None)
    # save_model_secs=60)

    # if not FLAGS.gpu:
    # config = tf.ConfigProto(device_count= {'GPU' : 0})
    # else:
    config = tf.ConfigProto()  # log_device_placement=True)
    config.gpu_options.allow_growth = True

    # print (FLAGS.gpu)

    # iter = []
    # train_loss = []
    # test_loss = []

    with sv.managed_session(server.target, config=config) as sess:
        if FLAGS.task_index == 0:
            print("Starting chief queue runner and running init_tokens_op")
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        step = 0
        start = time.time()
        local_step = 0
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test', sess.graph)
        
        logfile = open('dist.log','w')

        while not sv.should_stop() and step < FLAGS.max_steps:

            batch_idx = step % FLAGS.epoch_steps
            temp = feed_dict(True, batch_idx * FLAGS.num_workers + FLAGS.task_index)

            _, step = sess.run([train_step, global_step], feed_dict=temp)
            local_step += 1
            now = time.time()
            if step % 10 == 0:
                logfile.write("%.2f: Worker %d: training step %d done (global step: %d)\n" %
                      (now, FLAGS.task_index, local_step, step))
            if (step + 1) % FLAGS.epoch_steps == 0:  # or (step < FLAGS.epoch_steps and step %100 ==0):
                end = time.time()
                print("PureTrainTime: %-4.3fs" % (end - start))    
                epoch_loss = 0
                for i in range(FLAGS.epoch_steps):
                    loss_v = sess.run(loss,
                                      feed_dict=
                                      feed_dict(True, i * FLAGS.num_workers + FLAGS.task_index))
                    epoch_loss += loss_v

                epoch_loss /= FLAGS.epoch_steps
                #iter.append(1.0*step/FLAGS.epoch_steps)
                #train_loss.append(epoch_loss)
                train_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
                train_writer.add_summary(train_loss, step + 1)
                logfile.write("Epoch #%-2.3f | Train Loss: %-4.3f | PureTrainTime: %-3.3fs\n" %
                       (step / FLAGS.epoch_steps, epoch_loss, end - start))
                print ("Epoch #%-2.3f | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                       (step / FLAGS.epoch_steps, epoch_loss, end - start))

                epoch_loss = 0
                for i in range(FLAGS.epoch_steps):
                    loss_v = sess.run(loss,
                                      feed_dict=
                                      feed_dict(False, i * FLAGS.num_workers + FLAGS.task_index))
                    epoch_loss += loss_v

                epoch_loss /= FLAGS.epoch_steps
                start = time.time()
                #test_loss.append(epoch_loss)

                test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
                test_writer.add_summary(test_loss, step + 1)
                logfile.write("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs\n" %
                       (step / FLAGS.epoch_steps, epoch_loss, start - end))
                print("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
                       (step / FLAGS.epoch_steps, epoch_loss, start - end))
                # save_path = saver.save(sess, "/tmp/dssm-dist/model.ckpt", global_step=step / FLAGS.epoch_steps)
                # print("Epoch %d trained in %.2fs. Model saved in file: %s" % (
                #     step / FLAGS.epoch_steps, end - start, save_path))
                start = time.time()
        print("Worker %d done!\n" % FLAGS.task_index)

        sv.stop()

        # print("[")
        # for i in range(len(iter)):
        #     print("%2.2f, " % iter[i])
        # print("]\n[")
        # for i in range(len(iter)):
        #     print("%3.3f, " % train_loss[i])
        # print("]\n[")
        # for i in range(len(iter)):
        #     print("%3.3f, " % test_loss[i])
        # print("]")
