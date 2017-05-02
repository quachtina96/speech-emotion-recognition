#!/bin/bash

# for i in `seq 2 5`; do
	
	session_number=2
	utterance_dir="/home/quacht/speech-emotion-recognition/session/session${session_number}/"
	cd $utterance_dir
	for file in *; do
	utterance_path=${utterance_dir}$file
	echo $utterance_path
	sbatch /home/quacht/speech-remotion-recognition/extract_features.slurm $session_number $utterance_path
	done
# done
