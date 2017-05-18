#!/bin/sh
#SBATCH -n 1            # 16 cores
#SBATCH -t 1:00:00      # 1 day and 3 hours
#SBATCH -p tofu         # partition name
#SBATCH -J model        # sensible name for the job


# launch the code
python3 final_beast.py p s
python3 final_beast.py p g