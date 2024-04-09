#!/bin/bash
#SBATCH --job-name=deep_learning_job   # Job name
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task        
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=32G                      # Job memory request
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --output=job_logs/deep_learning_job_%j.log  # Standard output and error log
#SBATCH --gres=gpu:1                   # Request GPU resource

# Load any modules and activate your conda environment here
module spider python/3.10.2
module spider cuda/11.7
source /home/kka151/venvs/torch/bin/activate


# Navigate to your project directory (optional)




# Execute your deep learning script
python3 unet_merge.py
