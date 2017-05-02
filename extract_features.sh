#!/bin/bash

module load mit/matlab/2016b

matlab -nodesktop -nojvm < extract_experimental_features.m >> feature_extraction_log.txt
