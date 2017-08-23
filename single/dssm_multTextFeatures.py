#!/usr/bin/env python

from __future__ import print_function

import pickle
import random
import time
import sys
import numpy as np
import tensorflow as tf
import scipy as sp
import scipy.sparse
import csv
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

start = time.time()

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('test', 0, 'test flag')
flags.DEFINE_string('config_name', 'text_cat', 'The configuration to use')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.') 
flags.DEFINE_string('opt', 'sgd', 'Initial learning rate.') 
flags.DEFINE_integer('max_steps', 4000000, 'Number of steps to run trainer.') # max number of steps
flags.DEFINE_integer('epoch_steps', 88000, "Number of steps in one epoch.") # no. steps in epoch = (batches/pack * no. packs)
flags.DEFINE_integer('pack_size', 100, "Number of batches in one pickle pack.") # no. batches in each pack
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")
flags.DEFINE_integer('steps_per_eval', 10000, 'Interval between evaluation printouts.')
flags.DEFINE_integer('steps_per_save', 50000, 'Interval between writing scored files.')
flags.DEFINE_bool('USE_LAYER_3', 1, 'Whether to include a third layer or not')
flags.DEFINE_bool('USE_QUAD_DIST', 0, 'Use cosine distance metric')
flags.DEFINE_bool('USE_COS_DIST', 0, 'Use cosine distance metric')
flags.DEFINE_bool('USE_L2_DIST', 0, 'Use cosine distance metric')
flags.DEFINE_bool('USE_NN_LAYER', 0, 'Use cosine distance metric')
flags.DEFINE_string('scale_multiple', 'none', 'Whether to scale 2+ items in same ngram space')
flags.DEFINE_integer('L1_N', 400, 'The number of elements in the 1st layer')
flags.DEFINE_integer('L2_N', 400, 'The number of elements in the 2nd layer')
flags.DEFINE_integer('L3_N', 192, 'The number of elements in the 3nd layer')
flags.DEFINE_integer('L4_N', 192, 'The number of elements in the 4nd layer')
flags.DEFINE_integer('NEG', 50, 'The number of negative samples per positive')
flags.DEFINE_integer('BS', 1000, 'The batch-size')

dataset_constraints = {}
test = FLAGS.test
NEG = FLAGS.NEG 
BS = FLAGS.BS 
USE_LAYER_3 = FLAGS.USE_LAYER_3
USE_COS_DIST = FLAGS.USE_COS_DIST
USE_QUAD_DIST = FLAGS.USE_QUAD_DIST
USE_L2_DIST = FLAGS.USE_L2_DIST
USE_NN_LAYER = FLAGS.USE_NN_LAYER
L1_N = FLAGS.L1_N 
L2_N = FLAGS.L2_N 
L3_N = FLAGS.L3_N 
L4_N = FLAGS.L4_N 
scale_multiple = FLAGS.scale_multiple
opt = FLAGS.opt

config_description = "{}".format(FLAGS.config_name)
run_description = "{}_L{}-{}{}{}_{}NEG_{}".format(config_description, L1_N, L2_N, ('-'+str(L3_N) if USE_LAYER_3 else ''), ('-'+str(L4_N) if not USE_COS_DIST else ''), NEG, opt)

if scale_multiple != 'none':
    run_description += "_mult_scale_{}".format(scale_multiple)
if USE_QUAD_DIST:
    run_description += "_QUAD"
if USE_L2_DIST:
    run_description += "_L2"
if USE_NN_LAYER:
    run_description += "_NN"

summaries_dir = '/tmp/dssm_multTextFeatures/'+run_description
if test:
    run_description = "TEST__" + run_description
    summaries_dir = "/tmp/TEST"
  

print(run_description)
#sys.exit(1)


def calc_roc_auc(y, y_preds):
  fpr, tpr, _ = roc_curve(y, y_preds)
  roc_auc = auc(fpr, tpr)
  return roc_auc

def get_shuffled_index(M):
    index = np.arange(M.shape[0])
    np.random.shuffle(index)
    #return M[index, :]
    return index

def load_sparse_csr(filename):
    loader = np.load(filename + '.npz')
    return sp.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

def load_array(filename, val_type=int):
    return np.array([line.strip("\n") for line in open(filename)]).astype(val_type)


# ****************************** configuration for new dataset **************************************

# 1 step == 1 batch
batches_dataset_loc = "./data/batches"
spot_train_dataset_loc = "./data/train_dataset"
spot_test_dataset_loc = "/data/test_dataset"

batches_filename = "csr_100k_train_{}"
spot_filename = "csr_all"

datasets = {
    # text trigram BoW
    "text_trigram": {},
    "tokenized_text_trigram": {},
    "aspect_values_trigram": {},
    "cat_breadcrumb_trigram": {},

    # text BoW 
    "text_BoW": {},
    "tokenized_text_BoW": {},
    "aspect_values_BoW": {},
    "cat_breadcrumb_BoW": {},

    # category BoW
    "category_BoW": {},
    "category_leafcat_BoW": {}
}

csr_configurations = {   
    "text": {
        "text": ["text_trigram"]
    },
    "textBoW": {
        "textBoW": ["text_BoW"]
    },
    "text_leafcat": {
        "text": ["text_trigram"],
        "cat": ["category_leafcat_BoW"]
    },
    "text_cat": {
        "text": ["text_trigram"],
        "cat": ["category_BoW"]
    },
    "text_asp": {
        "text": ["text_trigram"],
        "asp": ["aspect_values_trigram"]
    },
    "text+asp": {
        "text": ["text_trigram", "aspect_values_trigram"]
    },
    "text_cat_asp": {
        "text": ["text_trigram"],
        "cat": ["category_BoW"],
        "asp": ["aspect_values_trigram"]
    },
    "text_leafcat_asp": {
        "text": ["text_trigram"],
        "cat": ["category_leafcat_BoW"],
        "asp": ["aspect_values_trigram"]
    },
    "text+asp+catBC": {
        "text": ["text_trigram", "aspect_values_trigram", "cat_breadcrumb_trigram"]
    },
    "text+asp+catBC_cat": {
        "text": ["text_trigram", "aspect_values_trigram", "cat_breadcrumb_trigram"],
        "cat": ["category_BoW"]
    },
    "text+asp_cat": {
        "text": ["text_trigram", "aspect_values_trigram"],
        "cat": ["category_BoW"]
    },
    "text_cat_aspBoW": {
        "text": ["text_trigram"],
        "cat": ["category_BoW"],
        "asp": ["aspect_values_BoW"]
    },
    "textTokenized_cat": {
        "text": ["tokenized_text_trigram"],
        "cat": ["category_BoW"]
    }
}
scale_multiples = {
    "none": [1.0, 1.0, 1.0, 1.0],
    "exp1": [1.0, 0.5, 0.25, 0.125],
    "exp2": [1.0, 0.1, 0.01, 0.001],
    "exp3": [1.0, 0.01, 0.0001, 0.000001],
    "linear1": [1.0, 0.8, 0.6, 0.4],
    "linear2": [1.0, 0.7, 0.4, 0.1],
    "inverted1": [1.0, 100.0, 10000.0, 1000000.0],
    "inverted2": [1.0, 10.0, 100.0, 1000.0],
    "inverted3": [1.0, 2.0, 3.0, 4.0],
    "random1": [1.0, 100.0, 10.0, 0.1],
    "random2": [1.0, 0.1, 100.0, 10.0]
}

csr_config = csr_configurations[FLAGS.config_name]

def get_csrs(dataset_path, filename, csr_config, dataset_constraints=None):
    d = {}
    for tp in csr_config:
        for i, ngram_type in enumerate(csr_config[tp]):
            query_csr = load_sparse_csr("{}/{}/query/{}".format(dataset_path, ngram_type, filename))
            item_csr = load_sparse_csr("{}/{}/item/{}".format(dataset_path, ngram_type, filename))
            
            # csr columns are sorted by frequency, so we can shed less useful columns on-demand
            if dataset_constraints is not None and tp in dataset_constraints:
                max_cols = dataset_constraints[tp]
                if max_cols < query_csr.shape[1]:
                    query_csr = query_csr[:,:max_cols]
                    item_csr = item_csr[:,:max_cols]
            
            if tp not in d:
                d[tp] = {"query": query_csr, "item": item_csr}
            else:
                if scale_multiple == 'none':
                    d[tp]["query"] += query_csr
                    d[tp]["item"] += item_csr
                else:
                    d[tp]["query"] += (query_csr * scale_multiples[scale_multiple][i])
                    d[tp]["item"] += (item_csr * scale_multiples[scale_multiple][i])



    M_query = scipy.sparse.hstack([d[tp]["query"] for tp in csr_config], format="csr")
    M_item = scipy.sparse.hstack([d[tp]["item"] for tp in csr_config], format="csr")
    
    return M_query, M_item

print("{} - loading spot train dataset for config {}".format(int(time.time()-start), FLAGS.config_name))
M_spot_train_query, M_spot_train_item = get_csrs(spot_train_dataset_loc, spot_filename, csr_config, dataset_constraints=dataset_constraints)
print("{} - spot train dataset loaded".format(int(time.time()-start)))

print("{} - loading spot test dataset for config {}".format(int(time.time()-start), FLAGS.config_name))
M_spot_test_query, M_spot_test_item = get_csrs(spot_test_dataset_loc, spot_filename, csr_config, dataset_constraints=dataset_constraints)
print("{} - spot test dataset loaded".format(int(time.time()-start)))

#***************************************************************************************************

TRIGRAM_D = M_spot_train_query.shape[1] 

doc_train_data = None
query_train_data = None

# load test data for now
print("{} - loading batch test dataset for config {}".format(int(time.time()-start), FLAGS.config_name))
query_test_data, doc_test_data = get_csrs(batches_dataset_loc, "csr_100k_test", csr_config, dataset_constraints=dataset_constraints)
print("{} - batch test dataset loaded".format(int(time.time()-start)))

idx = get_shuffled_index(query_test_data)
query_test_data, doc_test_data = query_test_data[idx,:], doc_test_data[idx,:]

datasets_to_score = [
    ("relevanceV6_train", "./data/relevance_train_submissions", {
        "query": M_spot_train_query,
        "title": M_spot_train_item,
        "label": load_array(spot_train_dataset_loc+"/label.tsv")
    }),
    ("relevanceV6_test", "./data/relevance_test_submissions", {
        "query": M_spot_test_query,
        "title": M_spot_test_item,
        "label": load_array(spot_test_dataset_loc+"/label.tsv")
    })
]

print("\n\n query size: {}, title size: {} \n\n".format(query_test_data.shape, doc_test_data.shape))

def load_train_data(pack_idx):
    global doc_train_data, query_train_data
    doc_train_data = None
    query_train_data = None
    start = time.time()
    query_train_data, doc_train_data = get_csrs(batches_dataset_loc, batches_filename.format(pack_idx), csr_config, dataset_constraints=dataset_constraints)
                                                

    #shuffle the dataset
    idx = get_shuffled_index(query_train_data)
    query_train_data, doc_train_data = query_train_data[idx,:], doc_train_data[idx,:]

    end = time.time()
    print("\nTrain data {} is loaded in {:.2f}".format(pack_idx, end - start))



end = time.time()
print("Loading data from HDD to memory: %.2fs" % (end - start))


with tf.name_scope('input'):
    query_batch = tf.sparse_placeholder(tf.float32, name='QueryBatch')
    doc_batch = tf.sparse_placeholder(tf.float32, name='DocBatch')

with tf.name_scope('L1'):
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
    weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range), name="weight1")
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range), name="bias1")

    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
    doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1) + bias1

    query_l1_out = tf.nn.relu(query_l1)
    doc_l1_out = tf.nn.relu(doc_l1)


if not USE_LAYER_3:
    with tf.name_scope('L2'):
        l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
        weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range), name="weight2")
        bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range), name="bias2")
    
        query_l2 = tf.matmul(query_l1_out, weight2) + bias2
        doc_l2 = tf.matmul(doc_l1_out, weight2) + bias2
    
        query_y = tf.nn.relu(query_l2, name="query_y")
        doc_y = tf.nn.relu(doc_l2, name="doc_y")

else:
    with tf.name_scope('L2'):
        l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))
        weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range), name="weight2")
        bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range), name="bias2")
    
        query_l2 = tf.matmul(query_l1_out, weight2) + bias2
        doc_l2 = tf.matmul(doc_l1_out, weight2) + bias2
    
        query_l2_out = tf.nn.relu(query_l2)
        doc_l2_out = tf.nn.relu(doc_l2)

    with tf.name_scope('L3'):
        l3_par_range = np.sqrt(6.0 / (L2_N + L3_N))
        weight3 = tf.Variable(tf.random_uniform([L2_N, L3_N], -l3_par_range, l3_par_range), name="weight3")
        bias3 = tf.Variable(tf.random_uniform([L3_N], -l3_par_range, l3_par_range), name="bias3")
    
        query_l3 = tf.matmul(query_l2_out, weight3) + bias3
        doc_l3 = tf.matmul(doc_l2_out, weight3) + bias3
    
        query_y = tf.nn.relu(query_l3, name="query_y")
        doc_y = tf.nn.relu(doc_l3, name="doc_y")


with tf.name_scope('FD_rotate'):
    # Rotate FD+ to produce 50 FD-
    temp = tf.tile(doc_y, [1, 1])
    print("doc_y shape: {}".format(doc_y.get_shape()))
    num_rows = tf.shape(doc_y)[0]

    for i in range(NEG):
        #GEN# rand = int((random.random() + i) * BS / NEG)
        rand = tf.to_int32((random.random() + i) * tf.to_float(num_rows) / NEG)
        doc_y = tf.concat([doc_y,
                           tf.slice(temp, [rand, 0], [num_rows - rand, -1]),
                           tf.slice(temp, [0, 0], [rand, -1])], 0)


if USE_QUAD_DIST:
    with tf.name_scope('Quadratic_similarity'):
        l4_par_range = np.sqrt(6.0 / (L3_N + L3_N))
        weight4 = tf.Variable(tf.random_uniform([L3_N, L3_N], -l4_par_range, l4_par_range), name="weight4")
        bias4 = tf.Variable(tf.random_uniform([1], -l4_par_range, l4_par_range), name="bias4")

        query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

        prod = tf.reduce_sum(tf.multiply(tf.matmul(doc_y, weight4), tf.tile(query_y, [NEG+1, 1])), 1, True)
        norm_prod = tf.multiply(query_norm, doc_norm)

        cos_sim_raw = tf.truediv(prod, norm_prod)
        raw_scores = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, -1])) 
elif USE_COS_DIST:
    with tf.name_scope('Cosine_Similarity'):
        # Cosine similarity
        query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))
    
        prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
        norm_prod = tf.multiply(query_norm, doc_norm)
    
        cos_sim_raw = tf.truediv(prod, norm_prod)
        raw_scores = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, -1])) 
elif USE_L2_DIST:
    with tf.name_scope('Euclid_dist'):
        prod = tf.sqrt(tf.reduce_sum(tf.square(tf.tile(query_y, [NEG + 1, 1]) - doc_y), 1, True))
        raw_scores = tf.transpose(tf.reshape(tf.transpose(prod), [NEG + 1, -1])) 

    
elif USE_NN_LAYER:
    # NN layer
    with tf.name_scope('L4'):
        l4_par_range = np.sqrt(6.0 / (2*L3_N + L4_N))
        weight4 = tf.Variable(tf.random_uniform([2*L3_N, L4_N], -l4_par_range, l4_par_range), name="weight4")
        bias4 = tf.Variable(tf.random_uniform([L4_N], -l4_par_range, l4_par_range), name="bias4")

        query_doc = tf.concat([doc_y, tf.tile(query_y, [NEG + 1, 1])], 1)
        query_doc_l4 = tf.matmul(query_doc, weight4) + bias4
        query_doc_l4_out = tf.nn.relu(query_doc_l4)

    
    with tf.name_scope('L5'):
        l5_par_range = np.sqrt(6.0 / (L4_N + 1))
        weight5 = tf.Variable(tf.random_uniform([L4_N, 1], -l5_par_range, l5_par_range), name="weight5")
        bias5 = tf.Variable(tf.random_uniform([1], -l5_par_range, l5_par_range), name="bias5")

        query_doc_l5 = tf.matmul(query_doc_l4_out, weight5) + bias5
        query_doc_l5_out = tf.nn.relu(query_doc_l5)

        raw_scores = tf.transpose(tf.reshape(tf.transpose(query_doc_l5_out), [NEG + 1, -1])) 

       
with tf.name_scope('Loss'):
    # Train Loss
    preds = tf.slice(raw_scores, [0, 0], [-1, 1])

    prob = tf.nn.softmax((raw_scores * 20))
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_mean(tf.log(hit_prob))
    tf.summary.scalar('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    if opt == 'ada' or opt == 'adadelta':
        train_step = tf.train.AdadeltaOptimizer(FLAGS.learning_rate).minimize(loss)
    elif opt == 'adam':
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    elif opt == 'sgd':
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)

with tf.name_scope('AucRoc'):
    roc_auc_loss = tf.placeholder(tf.float32)
    roc_auc_loss_summary = tf.summary.scalar('roc_auc_loss', roc_auc_loss)

def map_in_batches(query_csr, title_csr, sess, batch_size, feature):
    num_rows = query_csr.shape[0]
    res = np.zeros(num_rows)
    for start in range(0, num_rows, batch_size):
        end = min(start+batch_size, num_rows)
        query_csr_batch, title_csr_batch = query_csr[start:end], title_csr[start:end] 
        query_in, title_in = get_input_tensors(query_csr_batch, title_csr_batch)

        pred_scores = sess.run(feature, feed_dict={query_batch: query_in, doc_batch: title_in}).reshape(-1)
        res[start:end] = pred_scores
    return res

def get_input_tensors(query_in, doc_in):
    query_in = query_in.tocoo()
    doc_in = doc_in.tocoo()

    query_in = tf.SparseTensorValue(
        indices=np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        values=np.array(query_in.data, dtype=np.float),
        dense_shape=np.array(query_in.shape, dtype=np.int64))
    doc_in = tf.SparseTensorValue(
        indices=np.transpose([np.array(doc_in.row, dtype=np.int64), np.array(doc_in.col, dtype=np.int64)]),
        values=np.array(doc_in.data, dtype=np.float),
        dense_shape=np.array(doc_in.shape, dtype=np.int64))

    return query_in, doc_in

def pull_batch(query_data, doc_data, batch_idx):
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    return get_input_tensors(query_in, doc_in)


def feed_dict(Train, batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if Train:
        query_in, doc_in = pull_batch(query_train_data, doc_train_data, batch_idx)
    else:
        query_in, doc_in = pull_batch(query_test_data, doc_test_data, batch_idx)
    
    d = {query_batch: query_in, doc_batch: doc_in}
    return d


config = tf.ConfigProto()
# uses only as much GPU memory as needed
config.gpu_options.allow_growth=True

#with tf.Session() as sess:
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test', sess.graph)
    roc_auc_writer = tf.summary.FileWriter(summaries_dir + '/roc_auc', sess.graph)

    # Actual execution
    start = time.time()
    for step in range(FLAGS.max_steps):
        batch_idx = step % FLAGS.epoch_steps
        if batch_idx % FLAGS.pack_size == 0:
            load_train_data(batch_idx / FLAGS.pack_size + 1)

        if batch_idx % (FLAGS.pack_size / 64) == 0:
            progress = 100.0 * batch_idx / FLAGS.epoch_steps
            sys.stdout.write("\r{:.2f} Epoch (step {})".format(progress, step))
            sys.stdout.flush()

        sess.run(train_step, feed_dict=feed_dict(True, batch_idx % FLAGS.pack_size))

        if (step % FLAGS.steps_per_eval == 0 and step > 0) or (step+1 == FLAGS.max_steps):
            end = time.time()
            epoch_loss = 0
            for i in range(FLAGS.pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(True, i))
                epoch_loss += loss_v

            epoch_loss /= FLAGS.pack_size
            train_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(train_loss, step + 1)

            print ("\nStep #{:5} Epoch #{:5} | Train Loss: {:.3f} | PureTrainTime: {:.3f}s".format(step, step / FLAGS.epoch_steps, epoch_loss, end - start))

            epoch_loss = 0
            for i in range(FLAGS.pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(False, i))
                epoch_loss += loss_v

            epoch_loss /= FLAGS.pack_size

            test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            test_writer.add_summary(test_loss, step + 1)

            start = time.time()
            print ("\nStep #{:5} Epoch #{:5} | Train Loss: {:.3f} | PureTrainTime: {:.3f}s".format(step, step / FLAGS.epoch_steps, epoch_loss, start - end))

            roc_auc_scores = []
            for i, (dataset_name, dataset_path, d) in enumerate(datasets_to_score):
                scores = map_in_batches(d["query"], d["title"], sess, BS, preds)
                roc_auc_score = calc_roc_auc(d["label"], scores)
                print("\nscores:", scores[:20])
                print("{} - {:.5f}\n".format(dataset_name, roc_auc_score))


                if (step % FLAGS.steps_per_save == 0 and step > 0) or (step+1 == FLAGS.max_steps):
                    with open("{}/scored_{}_{}.tsv".format(dataset_path, run_description, step), "w") as out:
                        for n in scores:
                            print(n, file=out)

                if i == 0:
                    roc_auc_score = calc_roc_auc(d["label"], scores)
                    roc_auc_scores.append(roc_auc_score)


            avg_roc_auc_score = sess.run(roc_auc_loss_summary, feed_dict={roc_auc_loss: np.mean(roc_auc_scores)})
            roc_auc_writer.add_summary(avg_roc_auc_score, step + 1)




