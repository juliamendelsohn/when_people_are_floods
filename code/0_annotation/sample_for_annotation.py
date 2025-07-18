import numpy as np
import pandas as pd
import os
from pathlib import Path
from config import DATA_DIR, FULL_DATASET_FILE_WITH_SCORES, CONCEPTS


"""This script samples a subset of the full dataset for annotation.
It samples 200 examples for each concept, stratified by the GPT-4 heuristic scores.
The sampled data is saved in a directory named 'samples_to_annotate' within the DATA_DIR directory.
Not particularly useful for the public repo, but shared for transparency. 
"""


df = pd.read_csv(FULL_DATASET_FILE_WITH_SCORES,sep='\t')
out_dir = os.path.join(DATA_DIR, 'samples_to_annotate')
Path(out_dir).mkdir(parents=True, exist_ok=True)


# First strata is sample from no metaphors detected, remaining are evenly split among nonzero quantiles
# Heuristic scores are from the GPT-4 model, so we will sample from the gpt_ columns 
num_strata = 5 
num_samples_per_concept = 200
num_samples_per_stratum = int(num_samples_per_concept / num_strata)

for concept in CONCEPTS:
    strata_samples = []
    col = 'gpt_' + concept
    zero_subset = df[df[col] == 0].sample(n=num_samples_per_stratum)[['id_str','text']]
    zero_subset['stratum'] = 0
    strata_samples.append(zero_subset)
    nonzero_subset = df[df[col] > 0]
    nonzero_subset['quantile'] = pd.qcut(nonzero_subset[col],num_strata-1,labels=False)
    for i in range(num_strata-1):
        stratum = nonzero_subset[nonzero_subset['quantile'] == i]
        stratum_sample = stratum.sample(n=num_samples_per_stratum)[['id_str','text']]
        stratum_sample['stratum'] = i+1
        strata_samples.append(stratum_sample)
    concept_samples = pd.concat(strata_samples)
    out_file = os.path.join(out_dir,concept + '.tsv')
    concept_samples.to_csv(out_file,sep='\t',index=False)



