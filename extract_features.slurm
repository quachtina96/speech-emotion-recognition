#!/bin/bash


#SBATCH --job-name=extract
#SBATCH --time=24:00:00
#SBATCH -p sched_engaging_default

module load mit/matlab/2016b

session_number=$1
matlab -nodesktop -nojvm -r "extract_experimental_features $session_number" > session${session_number}_feature_extraction_log.txt
