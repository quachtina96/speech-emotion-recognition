#!/bin/bash

for i in `seq 1 5`; do
	
	session_number=$i
	utterance_dir="/home/quacht/speech-emotion-recognition/session/session${session_number}/"
	cd $utterance_dir
	for file in *; do
	utterance_path=${utterance_dir}$file
	python /home/quacht/speech-emotion-recognition/correct_utterance.py $utterance_path
	mv testing.txt $utterance_path
	echo $utterance_path
	done
done
