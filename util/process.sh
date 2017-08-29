date=20170828
set -x
rm -rf /data01/dssm/$date/query/
rm -rf /data01/dssm/$date/doc/
aws s3 cp --recursive s3://feednews-video-id/coldstart/dssm/clicktitle/monthly/$date/ /data01/dssm/$date/query/
aws s3 cp --recursive s3://feednews-video-id/coldstart/title/$date/ /data01/dssm/$date/doc/
cd /data01/dssm/$date/query/
cat part* > query_title
cd /data01/dssm/$date/doc/
cat part* > doc_title

cd /opt/dssm/
/usr/bin/python -m util.gen_vec --infile /data01/dssm/$date/query/query_title --outfile /data01/dssm/$date/query_vec --wordhash /data01/dssm/wordid --wordhashdim 9289
/usr/bin/python -m util.gen_vec --infile /data01/dssm/$date/doc/doc_title --outfile /data01/dssm/$date/doc_vec --wordhash /data01/dssm/wordid --wordhashdim 9289

cut -d$'\t' -f1 /data01/dssm/$date/doc_vec > /data01/dssm/$date/doc_vec.id
cut -d$'\t' -f2 /data01/dssm/$date/doc_vec > /data01/dssm/$date/doc_vec.vec
cut -d$'\t' -f1 /data01/dssm/$date/query_vec > /data01/dssm/$date/query_vec.id
cut -d$'\t' -f2 /data01/dssm/$date/query_vec > /data01/dssm/$date/query_vec.vec

python -m util.findknn --queryvec /data01/dssm/$date/query_vec.vec --queryid /data01/dssm/$date/query_vec.id --docvec /data01/dssm/$date/doc_vec.vec --docid /data01/dssm/$date/doc_vec.id --outfile /data01/dssm/$date/part.res.csv
