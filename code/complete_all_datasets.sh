BASE=../
DATA=datasets/
OUT=results/analysis_by_random_sample/

DATASET=CATNAP_Antibody_Mixtures
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --flat-file --obs-frac 1.0 --antibody-col-name Antibodies

DATASET=CATNAP_Monoclonal_Antibodies
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --flat-file --obs-frac 1.0

DATASET=CATNAP_Multispecific_Antibodies
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --flat-file --obs-frac 1.0

DATASET=CATNAP_Polyclonal_Sera
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --flat-file --obs-frac 1.0 --antibody-col-name Serum --value-name "ID50 (Neutralization Titer)" --data-transform log10

DATASET=Fonville2014_TableS1
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10

DATASET=Fonville2014_TableS13
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option concat
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option pre
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option post

DATASET=Fonville2014_TableS14
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option concat
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option pre
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option post

DATASET=Fonville2014_TableS3
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10

DATASET=Fonville2014_TableS5
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option concat
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option pre
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option post

DATASET=Fonville2014_TableS6
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option concat
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option pre
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --savepath $BASE/$OUT/$DATASET/ --obs-frac 1.0 --data-transform log10 --concat-option post