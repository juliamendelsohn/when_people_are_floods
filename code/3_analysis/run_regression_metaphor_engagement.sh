#!/bin/bash
#SBATCH --job-name=engagement
#SBATCH --account=juliame0
#SBATCH --partition=standard
#SBATCH --time=00-6:00:00
#SBATCH --mem=32gb
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/juliame/metaphor_public/code/analysis/
module load R

Rscript regression_metaphor_engagement.R 
Rscript regression_metaphor_engagement.R  --includeFrames 
Rscript regression_metaphor_engagement.R  --includeIdeology
Rscript regression_metaphor_engagement.R  --includeFrames --includeIdeology

