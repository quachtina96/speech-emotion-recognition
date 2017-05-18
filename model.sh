#!/bin/sh
#SBATCH -n 4            # 4 cores
#SBATCH -t 1:00:00      # 1 hours
#SBATCH -p sched_engaging_default         # partition name
#SBATCH -J tofu        # sensible name for the job


# launch the code
python3 final_beast.py p s
python3 final_beast.py p g