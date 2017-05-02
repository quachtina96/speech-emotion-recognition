#!/bin/bash

#SBATCH --job-name=extract
#SBATCH -p sched_engaging_default

module load mit/matlab/2016b

session_number=$1
utterance_path=$2
matlab -nodesktop -nojvm -r "extract_experimental_features $session_number $utterance_path";quit >> session${session_number}_feature_extraction_log.txt
