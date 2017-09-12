import sys
sys.path.append('/opt/faiss')
import faiss
import numpy as np
from sklearn.preprocessing import normalize
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--queryvec',type=str,help="input query vec file")
parser.add_argument('--queryid',type=str,help="input query id file")
parser.add_argument('--docvec',type=str,help="input doc vec file")
parser.add_argument('--docid',type=str,help="input doc id file")
parser.add_argument('--outfile',type=str,help="result knn file")
parser.add_argument('--knn',type=int,default=32,help="result knn file")
args = parser.parse_args()

userids = [x.strip("\n") for x in open(args.queryid)]
vids = [x.strip("\n") for x in open(args.docid)]
    
user_vec = np.loadtxt(open(args.queryvec,"rb"),delimiter=",",dtype=np.float32)
vid_vec = np.loadtxt(open(args.docvec,"rb"),delimiter=",",dtype=np.float32)
vid_vec_norm = np.array([x / np.linalg.norm(x) for x in vid_vec])
user_vec_norm = np.array([x / np.linalg.norm(x) for x in user_vec])
print 'dim ',vid_vec.shape[1]
print 'knn ',args.knn
index = faiss.IndexFlatIP(vid_vec_norm.shape[1])
index.train(vid_vec_norm)
index.add(vid_vec_norm)

dist,nn = index.search(user_vec_norm,args.knn)

outfile=open(args.outfile,"w")
for i in range(len(nn)):
    uid = userids[i]
    nn_lst = ";".join(["{}:{}".format(vids[nn[i][j]],dist[i][j]) for j in range(len(nn[i]))])
    outfile.write("{}\t{}\n".format(uid,nn_lst))
