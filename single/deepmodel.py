import tensorflow as tf
import numpy as np
import random
import yaml
from tqdm import tqdm
from query_doc import DataSet,Data
import sys
class Model:
    def __init__(self):
        self.paras = {
            'dssm':{
                'input_size':10*1000,
                'l1_size':256,
                'l2_size':128
            },
            'cf':{
                'embed_size': 300,
                'l1_size':256,
                'l2_size':128,
                'vid_num': 37652,
                'with_outputvid':True,#True: has output vid=>predict old videos. False: predict coldstart videos.
            },
            'imgdssm':{

            },
            'models':{
                'cf':True,
                'dssm':True,
                'img':False
            },
            'query_size':10,
            'batch_size':512,
            'negative_sample_size':50,
            'learning_rate':0.1,
            'num_steps':1000,
            'modeldir':'./model/',
            'logdir':'./logdir/',
            'traindat':'./data/train.txt',
            'testdat':'./data/test.txt'
        }
        self.query_img = None
        self.doc_img = None
        self.graph = None

    def config(self,configpath):
        with open(configpath, 'r') as f:
            content = f.read()
        self.paras = yaml.load(content)
        return self.paras

    def build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_cf_model()
            self.build_dssm_model()
            with tf.name_scope('agglayer'):
                input_x = []
                doc_y = []
                query_feat_size = 0
                doc_feat_size = 0
                if self.paras['models']['cf']:
                    print 'add cf input feature'
                    input_x.append(self.query_cf)
                    query_feat_size += self.paras['cf']['l2_size']
                    if self.paras['cf']['hasoutputvid']:
                        doc_y.append(self.doc_cf)
                        doc_feat_size += self.paras['cf']['l2_size']
                if self.paras['models']['dssm']:
                    print 'add dssm input feature'
                    input_x.append(self.query_dssm)
                    query_feat_size += self.paras['dssm']['l2_size']
                    doc_y.append(self.doc_dssm)
                    doc_feat_size += self.paras['dssm']['l2_size']

                if self.paras['models']['img']:
                    print 'add img input feature'
                    input_x.append(self.query_img)
                    doc_y.append(self.doc_img)
                    query_feat_size += self.paras['img']['l2_size']
                    doc_feat_size += self.paras['img']['l2_size']


                self.query = tf.concat(input_x,1) # query has batch_size*query_len rows
                self.doc = tf.concat(doc_y,1)

            with tf.name_scope('mix_layer'):
                query_mix_weights = tf.Variable(
                    tf.random_uniform([query_feat_size, query_feat_size/2], -1.0, 1.0),
                    name="query_mix_weights")
                query_mix_bias = tf.Variable(tf.zeros([query_feat_size/2]))

                doc_mix_weights = tf.Variable(
                    tf.random_uniform([doc_feat_size, query_feat_size/2], -1.0, 1.0),
                    name="query_mix_weights")
                doc_mix_bias = tf.Variable(tf.zeros([query_feat_size/2]))

                self.query = tf.matmul(self.query,query_mix_weights)+query_mix_bias
                self.query = tf.nn.tanh(self.query,name="query_vec")

                self.doc = tf.matmul(self.doc,doc_mix_weights)+doc_mix_bias
                self.doc = tf.nn.tanh(self.doc,name="doc_vec")


            with tf.name_scope('sample_negative'):
                temp = tf.tile(self.doc, [1, 1])  # temp equals doc_y here. just a replica
                for i in range(self.paras['negative_sample_size']):
                    rand = int((random.random()+i)*self.paras['batch_size']/self.paras['negative_sample_size'])
                    self.doc = tf.concat([self.doc,
                                          tf.slice(temp,[rand,0],[self.paras['batch_size'] - rand,-1]),
                                          tf.slice(temp,[0,0],[rand,-1])],0)
                    # each time the temp(replicated from doc_y) tensor will be splitted into 2 parts randomly(the split point is randomly choosen) and concated inversely
                    # Simple Example: [1,2,3,4,5] => [4,5] [1,2,3]=>[4,5,1,2,3]
                    # At the end loop doc_y becomes a (NEG+1)*BatchSize rows of vectors

            with tf.name_scope('cosine'):
                query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.query),1,True)),[self.paras['negative_sample_size']+1,1])
                doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc),1,True))
                prod = tf.reduce_sum(tf.multiply(tf.tile(self.query,[self.paras['negative_sample_size']+1,1]),self.doc),1,True)
                norm_prod = tf.multiply(query_norm,doc_norm)
                cos_sim_raw = tf.truediv(prod,norm_prod)
                cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [self.paras['negative_sample_size'] + 1, self.paras['batch_size']])) * 20

            with tf.name_scope('loss'):
                prob = tf.nn.softmax(cos_sim)
                hit_prob = tf.slice(prob,[0,0],[-1,1])
                self.loss = -tf.reduce_sum(tf.log(hit_prob))/self.paras['batch_size']

            with tf.name_scope('training'):
                self.train_step = tf.train.AdagradOptimizer(self.paras['learning_rate']).minimize(self.loss)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(prob,1),0)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
                tf.summary.scalar('accuracy',self.accuracy)

            with tf.name_scope('evaluation'):
                self.eval_loss = tf.placeholder(tf.float32)
                self.eval_acc = tf.placeholder(tf.float32)
                self.loss_summary = tf.summary.scalar('loss',self.eval_loss)
                self.acc_summary = tf.summary.scalar('acc',self.eval_acc)

    def build_cf_model(self):
        with tf.name_scope('cf'):
            with tf.name_scope('input'):
                self.query_vid_input = tf.placeholder(tf.int32,[None,self.paras['query_size']],name="query_vid")
                input_batch_size = tf.shape(self.query_vid_input)[0]
                query_vid_input = tf.reshape(self.query_vid_input,[input_batch_size*self.paras['query_size']])
                self.doc_vid = tf.placeholder(tf.int32,[None,1],name="doc_vid")
                input_batch_size = tf.shape(self.doc_vid)[0]
                doc_vid = tf.reshape(self.doc_vid,[input_batch_size])
            with tf.name_scope('embedding'):
                vid_embedding = tf.Variable(tf.random_uniform([self.paras['cf']['vid_num'],self.paras['cf']['embed_size']],-1.0,1.0),name="embedding")
                query_vid_embed = tf.nn.embedding_lookup(vid_embedding,query_vid_input)
                doc_vid_embed = tf.nn.embedding_lookup(vid_embedding,doc_vid)
            with tf.name_scope('L1'):
                l1_par_range = np.sqrt(6.0 / (self.paras['cf']['embed_size'] + self.paras['cf']['l1_size']))
                weight_l1 = tf.Variable(
                    tf.random_uniform([self.paras['cf']['embed_size'], self.paras['cf']['l1_size']], -l1_par_range,
                                      l1_par_range))
                bias_l1 = tf.Variable(tf.random_uniform([self.paras['cf']['l1_size']], -l1_par_range, l1_par_range))
                query_l1 = tf.matmul(query_vid_embed, weight_l1) + bias_l1
                doc_l1 = tf.matmul(doc_vid_embed, weight_l1) + bias_l1
                query_l1_out = tf.nn.relu(query_l1)
                doc_l1_out = tf.nn.relu(doc_l1)
            with tf.name_scope('L2'):
                l2_par_range = np.sqrt(6.0 / (self.paras['cf']['l1_size'] + self.paras['cf']['l2_size']))
                weight_l2 = tf.Variable(
                    tf.random_uniform([self.paras['cf']['l1_size'], self.paras['cf']['l2_size']], -l2_par_range,
                                      l2_par_range))
                bias_l2 = tf.Variable(tf.random_uniform([self.paras['cf']['l2_size']], -l2_par_range, l2_par_range))
                query_l2 = tf.matmul(query_l1_out, weight_l2) + bias_l2
                doc_l2 = tf.matmul(doc_l1_out, weight_l2) + bias_l2
                query_cf = tf.nn.tanh(query_l2, name="query_cf")
                self.doc_cf = tf.nn.tanh(doc_l2, name="doc_cf")
            with tf.name_scope('avg_query_cf'):
                query_cf = tf.reshape(query_cf,[self.paras['batch_size'],self.paras['query_size'],self.paras['cf']['l2_size']])
                self.query_cf = tf.reduce_mean(query_cf,axis=1,name="avg_query_cf")


    def build_dssm_model(self):
        with tf.name_scope('dssm'):
            with tf.name_scope('input'):
                self.query_dssm_input = tf.sparse_placeholder(tf.float32,[None,self.paras['dssm']['input_size']],name="dssm_query")
                self.doc_dssm_input = tf.sparse_placeholder(tf.float32,[None,self.paras['dssm']['input_size']],name="dssm_doc")
            with tf.name_scope('L1'):
                l1_par_range = np.sqrt(6.0/(self.paras['dssm']['input_size']+self.paras['dssm']['l1_size']))
                weight_l1 = tf.Variable(tf.random_uniform([self.paras['dssm']['input_size'],self.paras['dssm']['l1_size']],-l1_par_range,l1_par_range))
                bias_l1 = tf.Variable(tf.random_uniform([self.paras['dssm']['l1_size']],-l1_par_range,l1_par_range))
                query_l1 = tf.sparse_tensor_dense_matmul(self.query_dssm_input,weight_l1) + bias_l1
                doc_l1 = tf.sparse_tensor_dense_matmul(self.doc_dssm_input,weight_l1)+bias_l1
                query_l1_out = tf.nn.relu(query_l1)
                doc_l1_out = tf.nn.relu(doc_l1)
            with tf.name_scope('L2'):
                l2_par_range = np.sqrt(6.0/(self.paras['dssm']['l1_size']+self.paras['dssm']['l2_size']))
                weight_l2 = tf.Variable(tf.random_uniform([self.paras['dssm']['l1_size'],self.paras['dssm']['l2_size']],-l2_par_range,l2_par_range))
                bias_l2 = tf.Variable(tf.random_uniform([self.paras['dssm']['l2_size']],-l2_par_range,l2_par_range))
                query_l2 = tf.matmul(query_l1_out,weight_l2)+bias_l2
                doc_l2 = tf.matmul(doc_l1_out,weight_l2)+bias_l2
                query_dssm = tf.nn.tanh(query_l2,name="query_dssm")
                self.doc_dssm = tf.nn.tanh(doc_l2,name="doc_dssm")
                print 'self.doc_dssm shape: ',tf.shape(self.doc_dssm)
            with tf.name_scope('AvgQueryDSSM'):
                query_dssm = tf.reshape(query_dssm,[self.paras['batch_size'],self.paras['query_size'],self.paras['dssm']['l2_size']])
                print 'query_dssm shape: ', tf.shape(query_dssm)
                self.query_dssm = tf.reduce_mean(query_dssm,axis=1,name="avg_query_dssm")
                print 'self.query_dssm shape: ',tf.shape(self.query_dssm)



    def build_img_model(self):
        pass

    def train(self,training_set,test_set):
        batch_size = self.paras['batch_size']
        batch_num = training_set.size()/batch_size
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True        

        with tf.Session(graph=self.graph,config=config) as sess:
            tf.initialize_all_variables().run()
            train_writer = tf.summary.FileWriter(self.paras['logdir'] + 'train', self.graph)
            test_writer = tf.summary.FileWriter(self.paras['logdir'] + 'test', self.graph)
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=self.paras['num_steps'])
            for step in range(self.paras['num_steps']):
                #Training
                for batch_idx in tqdm(range(batch_num)):
                    input_vid,input_title,output_vid,output_title = training_set.get_data(batch_idx*batch_size,
                                                                                          batch_size,
                                                                                          self.paras['dssm']['input_size'])
                    sess.run(self.train_step,feed_dict={self.query_vid_input:input_vid,
                                                        self.query_dssm_input:input_title,
                                                        self.doc_vid:output_vid,
                                                        self.doc_dssm_input:output_title})
                #Evaluation on Training Set
                epoch_loss = 0.0
                epoch_acc = 0.0
                for batch_idx in tqdm(range(batch_num)):
                    input_vid, input_title, output_vid, output_title = training_set.get_data(batch_idx * batch_size,
                                                                                         batch_size,
                                                                                        self.paras['dssm']['input_size'])
                    loss,acc = sess.run([self.loss,self.accuracy],feed_dict={self.query_vid_input:input_vid,
                                                        self.query_dssm_input:input_title,
                                                        self.doc_vid:output_vid,
                                                        self.doc_dssm_input:output_title})
                    epoch_loss += loss
                    epoch_acc += acc
                epoch_loss /= batch_num
                epoch_acc /= batch_num

                train_loss,train_acc = sess.run([self.loss_summary,self.acc_summary],
                                                feed_dict={self.eval_loss:epoch_loss,self.eval_acc:epoch_acc})
                train_writer.add_summary(train_loss,step+1)
                train_writer.add_summary(train_acc,step+1)

                #Evaluation on Test Set
                epoch_loss = 0.0
                epoch_acc = 0.0
                for batch_idx in tqdm(range(test_set.size()/batch_size)):
                    input_vid, input_title, output_vid, output_title \
                        = test_set.get_data(batch_idx * batch_size,batch_size,self.paras['dssm']['input_size'])
                    loss,acc = sess.run([self.loss,self.accuracy],feed_dict={self.query_vid_input:input_vid,
                                                        self.query_dssm_input:input_title,
                                                        self.doc_vid:output_vid,
                                                        self.doc_dssm_input:output_title})
                    epoch_acc += acc
                    epoch_loss += loss
                epoch_loss /= batch_num
                epoch_acc /= batch_num
                test_loss,test_acc = sess.run([self.loss_summary,self.acc_summary],
                                              feed_dict={self.eval_loss:epoch_loss,self.eval_acc:epoch_acc})
                test_writer.add_summary(test_loss,step+1)
                test_writer.add_summary(test_acc,step+1)
                saver.save(sess,self.paras['modeldir'],global_step=step)



if __name__ == '__main__':
    confpath = sys.argv[1]
    model = Model()
    model.config(confpath)
    model.build_model()

    training_set = DataSet()
    training_set.load(model.paras['traindat'])
    test_set = DataSet()
    test_set.load(model.paras['testdat'])

    model.train(training_set,test_set)


