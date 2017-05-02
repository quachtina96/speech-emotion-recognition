#!/bin/bash

for i in `seq 1 5`; do
	sbatch extract_features.slurm $i
done
