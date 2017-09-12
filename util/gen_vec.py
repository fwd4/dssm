#input format: id TAB title
#output format: id vec
import tensorflow as tf
import numpy as np
from single import datautil
import preparedssm
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--infile',type=str,help="input text file",default="/opt/dssm/data/data.0")
parser.add_argument('--outfile',type=str,help="result vec file",default="/opt/dssm/data/queryvec.out")
parser.add_argument('--modelpath',type=str)
parser.add_argument('--wordhash',type=str,default="/opt/dssm/data/wordid")
parser.add_argument('--wordhashdim',type=int,help="word hash dimension")
parser.add_argument('--vectype',type=str,help="specify query vector or doc vector",default="queryvec")
args = parser.parse_args()

def load(inpath,dictpath):
    word_id_table = preparedssm.load_dict(dictpath)
    dataids = []
    sparse_vecs = []
    with open(inpath) as infile:
        for line in infile:
            pars = line.strip("\n").split('\t')
            if len(pars)<2: continue
            dataid = pars[0]
            title = pars[1]
            trigrams = preparedssm.title2trigams(title)
            encoded_ids = [word_id_table[x] for x in trigrams if x in word_id_table]
            sparseVec = datautil.SparseVector()
            sparseVec.indices = encoded_ids
            sparseVec.values = [1.0]*len(encoded_ids)
            sparse_vecs.append(sparseVec)
            dataids.append(dataid)
    return dataids,sparse_vecs

if __name__ == '__main__':
    ids,vecs = load(args.infile,args.wordhash)
    outfile = open(args.outfile,mode='w')
    checkpoint_file = args.modelpath#tf.train.latest_checkpoint(args.modeldir)
    #checkpoint_file = "/opt/dssm/model/20170828/-8314"
    batch_size = 512
    print '################checkpoint_file \n',checkpoint_file
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            #print graph.get_operations() #'input/QueryBatch/shape' type=Placeholder>, <tf.Operation 'input/QueryBatch/values' type=Placeholder>, <tf.Operation 'input/QueryBatch/indices'
            query_batch_shape = graph.get_operation_by_name("input/QueryBatch/shape").outputs[0]
            query_batch_values = graph.get_operation_by_name("input/QueryBatch/values").outputs[0]
            query_batch_indices = graph.get_operation_by_name("input/QueryBatch/indices").outputs[0]

            doc_batch_shape = graph.get_operation_by_name("input/DocBatch/shape").outputs[0]
            doc_batch_values = graph.get_operation_by_name("input/DocBatch/values").outputs[0]
            doc_batch_indices = graph.get_operation_by_name("input/DocBatch/indices").outputs[0]

            #doc_batch = graph.get_operation_by_name("input/DocBatch:0")
            #query_vec = graph.get_operation_by_name("L2/query_vec").outputs[0]
            query_vec = graph.get_operation_by_name("L2/query_vec").outputs[0]            
            if args.vectype=="docvec":
                query_vec = graph.get_operation_by_name("L2/doc_vec").outputs[0]
            
            #doc_vec = graph.get_operation_by_name("L2/doc_vec").outputs[0]
            group_num = len(ids) / batch_size - 1
            print 'generate soft max result for each sub sequence'
            print 'group num ', group_num
            for i in tqdm(range(group_num)):
                stv = datautil.TrainingData.toSparseTensorValue(vecs[i * batch_size:(i + 1) * batch_size],args.wordhashdim)
                #print 'stv ',stv
                #print 'shape, ',stv.dense_shape
                if args.vectype=="queryvec":
                    predict_res_set = sess.run(query_vec, {query_batch_shape: stv.dense_shape,query_batch_values:stv.values,query_batch_indices:stv.indices})
                else:                    
                    predict_res_set = sess.run(query_vec, {doc_batch_shape: stv.dense_shape,doc_batch_values:stv.values,doc_batch_indices:stv.indices})
                id_set = ids[i*batch_size:(i+1)*batch_size]
                for j in  range(batch_size):
                    res_line = "{}\t{}\n".format(id_set[j],",".join([str(x) for x in predict_res_set[j]]))
                    outfile.write(res_line)
    outfile.close()
