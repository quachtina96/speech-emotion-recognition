#!/bin/bash
#SBATCH --job-name=extract
#SBATCH -p sched_engaging_default

# This bash script, extract_features.slurm, ias written to be run on the 
# engaging cluster.

# load MATLAB 2016b module
module load mit/matlab/2016b

# Pull the first and second arguments supplied to the script to set the session 
# number and the path to single list of utterances
session_number=$1
utterance_path=$2

old_dir=$(pwd)
cd /home/quacht/speech-emotion-recognition/
# Pass these arguments to the MATLAB function extract_experimental_features in 
# order to extract features from the wav files indicated by the utterance list 
# and save them to 3 .csv files per utterance. 
# NOTE: that the when this function is called, the MATLAB is launched and does not quit.... I think.
matlab -nodesktop -nojvm -r "extract_experimental_features $session_number $utterance_path;quit" # >> session${session_number}_feature_extraction_log.txt

cd $old_dir
