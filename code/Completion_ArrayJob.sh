#! /bin/bash

#$ -cwd
#$ -N Completion
#$ -t 1-475
#$ -q broad
#$ -P regevlab
#$ -l h_vmem=10g
#$ -l h_rt=47:00:00
#$ -e Logs/
#$ -o Logs/

sleep $((SGE_TASK_ID%60))
source /broad/software/scripts/useuse
source /home/unix/bcleary/.my.bashrc
export OMP_NUM_THREADS=1

BASE=../
DATA=datasets/
OUT=results/individual_trials/
read p <<< $(sed -n ${SGE_TASK_ID}p params.txt)

DATASET=CATNAP_Antibody_Mixtures
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --flat-file --obs-frac $p --antibody-col-name Antibodies

DATASET=CATNAP_Monoclonal_Antibodies
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --flat-file --obs-frac $p

DATASET=CATNAP_Multispecific_Antibodies
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --flat-file --obs-frac $p

DATASET=CATNAP_Polyclonal_Sera
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --flat-file --obs-frac $p --antibody-col-name Serum --value-name "ID50 (Neutralization Titer)" --data-transform log10

DATASET=Fonville2014_TableS1
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10

DATASET=Fonville2014_TableS13
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option concat
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option pre
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option post

DATASET=Fonville2014_TableS14
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option concat
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option pre
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option post

DATASET=Fonville2014_TableS3
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10

DATASET=Fonville2014_TableS5
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option concat
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option pre
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option post

DATASET=Fonville2014_TableS6
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option concat
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option pre
python completion_job.py --dataset $BASE/$DATA/$DATASET.csv --job-id ${SGE_TASK_ID} --savepath $BASE/$OUT/$DATASET/ --obs-frac $p --data-transform log10 --concat-option post
