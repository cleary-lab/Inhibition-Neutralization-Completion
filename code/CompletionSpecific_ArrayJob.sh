#! /bin/bash

#$ -cwd
#$ -N Completion
#$ -t 1-311
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
OUT=results/specific_masks/

DATASET=$BASE/$DATA/Fonville2014_TableS1.csv,$BASE/$DATA/Fonville2014_TableS3.csv,$BASE/$DATA/Fonville2014_TableS5.csv,$BASE/$DATA/Fonville2014_TableS6.csv,$BASE/$DATA/Fonville2014_TableS13.csv,$BASE/$DATA/Fonville2014_TableS14.csv
python completion_job.py --dataset $DATASET --savepath $BASE/$OUT/AllFonnville --specific-mask --data-transform log10 --concat-option concat --job-id ${SGE_TASK_ID} --virus-table-path $BASE/$DATA/Fonville_viruses.csv

