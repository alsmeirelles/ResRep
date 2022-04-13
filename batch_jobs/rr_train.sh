#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --exclude=v034
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

DIRID="RR-12"
cd /ocean/projects/asc130006p/alsm/ResRep

echo '[VIRTUALENV]'
source venv/bin/activate

#Load CUDA and set LD_LIBRARY_PATH
module load cuda/10.0.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ocean/projects/asc130006p/alsm/venv/lib64/cuda-10.0.0
export CUDA_VISIBLE_DEVICES=0

echo '[START] training'
date +"%D %T"

time python3 alsm.py --rr --pred -net swresnet50v2 -data ../active-learning/data/AL-367 -test ../active-learning/data/Test-T8 -e 180 -aug -b 64 -tdim 240 240 -cpu 2 -gpu 1 -lr 0.0005 -eval 7 -out results/$DIRID 

echo '[FINAL] done training'

deactivate 

date +"%D %T"


