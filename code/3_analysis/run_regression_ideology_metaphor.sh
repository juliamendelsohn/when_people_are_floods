#!/bin/bash
#SBATCH --job-name=regression1
#SBATCH --account=juliame0
#SBATCH --partition=standard
#SBATCH --time=00-12:00:00
#SBATCH --mem=32gb
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/juliame/metaphor_public/code/analysis/
module load R
Rscript regression_ideology_metaphor.R 
Rscript regression_ideology_metaphor.R  --includeFrames 
