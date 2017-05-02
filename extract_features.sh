#!/bin/bash
#SBATCH --job-name=extract1
#SBATCH --output=res1.txt
#SBATCH -time=0   # no limit on time
#SBATCH -p sched_engaging_default      # partition name

module load mit/matlab/2016b

matlab -nodesktop -nojvm < extract_experimental_features.m >> feature_extraction_log.txt
