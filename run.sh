#!/bin/sh 


#SBATCH --job-name=chexficient
#SBATCH -A marlowe-m000081
#SBATCH -p preempt


#SBATCH --nodes=1
#SBATCH -G 8
#SBATCH --time=64:00:00
##SBATCH --error=~/foo.err
##SBATCH --nodelist=n01


#SBATCH --mem=500G
#SBATCH --cpus-per-task=24


module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75


conda activate /projects/m000081/ppn


python -m main