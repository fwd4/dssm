import re
import sys
id_table = {} #map trigram to trigram ID
query_id = {}
doc_id = {}
def title2trigams(title):
    title = title.lower()
    words = "".join([x for x in title if ((x>='A' and x<='Z') or (x>='a' and x<='z') or x==' ')])
    words = words.split()
    res = []
    for word in words:
        padded_word = "#"+word+"#"
        tri_tuples = zip(padded_word, padded_word[1:], padded_word[2:])
        seq = ["".join(x) for x in tri_tuples]
        res.extend(seq)
    return res

def encode(terms):
    ids = []
    for term in terms:
        if term in id_table:
            ids.append(id_table[term])
        else:
            term_id = len(id_table)
            id_table[term] = term_id
            ids.append(term_id)
    return ids


def encode_wordhash(inpath,outpath,id_dict):
    with open(inpath) as infile, open(outpath, 'w') as outfile:
        for line in infile:
            line = line.strip("\n")
            pars = line.split("\t")
            id_ = pars[0]
            if id_ not in id_dict:
                id_dict[id_] = len(id_dict)
            wordhash = encode(title2trigams(line))
            if len(wordhash) == 0:
                continue
            res_str = ' '.join(["{}:1.0".format(str(x)) for x in wordhash])
            outfile.write('{}\n'.format(res_str))

def save_dict(id_table,path):
    dictfile = open(path,'w')
    for kv in id_table.iteritems():
        dictfile.write('{} {}\n'.format(kv[0],kv[1]))
    dictfile.close()

import operator
def parse_seq(inpath,query_out,doc_out,wordid_out):
    with open(inpath) as infile,open(query_out,"w") as queryfile,open(doc_out,"w") as docfile:
        for line in infile:
            click_ts = {}
            pars = line.strip(" \"\n").split("\t")
            if len(pars)<2:
                continue
            uid = pars[0]
            seq = pars[1].split(";")
            for title_ts in seq:
                if ":" not in title_ts:
                    continue
                fields = title_ts.split(":")
                click_ts[fields[0]] = int(fields[1])
            if len(click_ts)<=1:
                continue
            sorted_ts = sorted(click_ts.items(),key=operator.itemgetter(1))
            watched_titles = sorted_ts[:-1]
            next_title = sorted_ts[-1][0]

            input_title = " ".join([x[0] for x in watched_titles])
            wordhash = encode(title2trigams(input_title))
            if len(wordhash) == 0:
                continue
            res_str = ' '.join(["{}:1.0".format(str(x)) for x in wordhash])
            queryfile.write(res_str+"\n")

            wordhash = encode(title2trigams(next_title))
            if len(wordhash) == 0:
                continue
            res_str = ' '.join(["{}:1.0".format(str(x)) for x in wordhash])
            docfile.write(res_str+"\n")
    save_dict(id_table,wordid_out)

def seperate_process():
    query_inpath = sys.argv[1]
    doc_inpath = sys.argv[2]
    query_outpath = query_inpath+".out"
    doc_outpath = doc_inpath+".out"
    worddictpath = sys.argv[3]

    encode_wordhash(query_inpath,query_outpath,query_id)
    encode_wordhash(doc_inpath,doc_outpath,doc_id)

    save_dict(id_table,worddictpath)

    save_dict(query_id,query_inpath+".queryid")
    save_dict(doc_id,doc_inpath+".docid")

if __name__ == '__main__':
    parse_seq(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])




