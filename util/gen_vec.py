#input format: id TAB title
#output format: id vec
import tensorflow as tf
import numpy as np
from single import datautil
import preparedssm
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--inputfile',type=str,help="input text file",default="/opt/seqmodel/data/20170706/seq.id.txt")
parser.add_argument('--outfile',type=str,help="result vec file")
parser.add_argument('--modeldir',type=str,default="/opt/dssm/model/")
parser.add_argument('--wordhash',type=str,default="/opt/dssm/data/worddict.txt")
args = parser.parse_args()

def load(inpath,dictpath):
    word_id_table = preparedssm.load_dict(dictpath)
    dataids = []
    sparse_vecs = []
    with open(inpath) as infile:
        for line in infile:
            pars = line.strip("\n").split("\t")
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
    ids,vecs = load(args.inputfile,args.wordhash)
    outfile = open(args.outfile,mode='w')
    checkpoint_file = tf.train.latest_checkpoint(args.modeldir)
    batch_size = 512
    print '################checkpoint_file \n',checkpoint_file
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            query_batch = graph.get_operation_by_name("input/QueryBatch")
            doc_batch = graph.get_operation_by_name("input/DocBatch")
            query_vec = graph.get_operation_by_name("L2/query_vec").outputs[0]
            #doc_vec = graph.get_operation_by_name("L2/doc_vec").outputs[0]
            group_num = len(ids) / batch_size - 1
            print 'generate soft max result for each sub sequence'
            print 'group num ', group_num
            for i in tqdm(range(group_num)):
                sparseTensorValue = datautil.TrainingData.toSparseTensorValue(vecs[i * batch_size:(i + 1) * batch_size])
                predict_res_set = sess.run(query_vec, {query_batch: sparseTensorValue})
                id_set = ids[i*batch_size:(i+1)*batch_size]
                for j in  range(batch_size):
                    res_line = "{}\t{}\n".format(id_set[j],",".join([str(x) for x in query_vec]))
                    outfile.write(res_line)
    outfile.close()
