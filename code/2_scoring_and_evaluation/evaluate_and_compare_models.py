import pandas as pd 
import os
import json
import simpledorff
from itertools import combinations
from sklearn import metrics
import warnings
from collections import Counter
from pathlib import Path
import scipy
import numpy as np
import argparse

import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from config import METAPHOR_SCORES_DIR, ANNOTATED_DATA_FILE, CONCEPTS, LABEL_COL, EVALUATION_DIR, AUC_DEFAULT_THRESHOLD

""" This script evaluates a single metaphor scoring model against human annotations. 
The input file name is "{METAPHOR_SCORES_DIR}_for_eval/{embedding_model_name}_{llm}_{prompt}/scores_no_text.tsv".
The input file contains scores for each concept in columns named 'score_{concept}' (combined),
'sbert_{concept}' (SBERT scores), and 'llm_{concept}' (LLM scores). 

This file is meant to calculate and compare all models' scores against human annotations and with each other.
Depending on your use case, it may be more useful to take some of these functions separately rather than running the whole script.
"""


def load_ground_truth_annotations(annotated_data_file, label_col):
    # load annotated data file with id_str, concept, and label_col
    df_annots = pd.read_csv(annotated_data_file, sep='\t', dtype=str)
    df_annots = df_annots[['id_str', 'concept', label_col]]
    df_annots = df_annots.rename(columns={label_col: 'y_true'})
    return df_annots


# Load the model scores. Default will be combined score ("score"), but you can specify 'sbert' or 'llm' to load those scores instead.
def load_model_scores(scores_file, model_name, concepts, mode = 'score'):
    df_scores = pd.read_csv(scores_file, sep='\t', dtype=str)
    concept_cols = [f'{mode}_{concept}' for concept in concepts]
    df_scores = df_scores[['id_str'] + concept_cols]
    df_scores.columns = [col.replace(f'{mode}_', '') for col in df_scores.columns]
    # Now, the structure is id_str, concept1, concept2, ..., conceptN with values being metaphor scores. 
    # Convert to long format
    df_scores = pd.melt(df_scores, id_vars='id_str', var_name='concept', value_name='y_pred')
    df_scores['y_pred'] = df_scores['y_pred'].astype(float)
    df_scores['model'] = model_name
    return df_scores

def load_all_model_scores(metaphor_scores_dir, concepts, models, mode = 'score',filename='scores_no_text.tsv'):
    df_all_scores = []
    for model in models:
        scores_file = os.path.join(metaphor_scores_dir, model, filename)
        if os.path.exists(scores_file):
            df_scores = load_model_scores(scores_file, model, concepts, mode=mode)
            df_all_scores.append(df_scores)
    df_all_scores = pd.concat(df_all_scores, ignore_index=True)
    return df_all_scores

def combine_annotations_and_model_scores(df_annots, df_scores):
    # Merge the annotations and scores on id_str and concept
    df_combined = df_annots.merge(df_scores, on=['id_str', 'concept'])
    # Drop duplicates if any
    df_combined = df_combined.drop_duplicates()
    return df_combined

# After loading the annotations and model scores, we have a single dataframe with columns:
# id_str, concept, y_true, y_pred, model
# By default, calculates score for all concepts, but you can specify a single concept to evaluate.
def evaluate_model_spearman(df,model, concept=None,exclude_domain_agnostic=True):
    df_sub = df[df['model']==model]
    if exclude_domain_agnostic:
        df_sub = df_sub[df_sub['concept'] != 'domain-agnostic']
    if concept:
        df_sub = df_sub[df_sub['concept']==concept]
    y_true = df_sub['y_true']
    y_pred = df_sub['y_pred']
    # Compare spearman rank between y_true and y_pred
    n = len(y_pred)
    r, p = scipy.stats.spearmanr(y_true,y_pred)
    return n,r,p

def evaluate_all_models_spearman(df,concept=None,exclude_domain_agnostic=True):
    models = df['model'].unique()
    results = []
    for model in models:
        n, r, p = evaluate_model_spearman(df,model,concept,exclude_domain_agnostic)
        concept_name = concept if concept else 'all_documents'
        results.append({'model':model,'concept':concept_name,'r':r})
    results = pd.DataFrame(results)
    return results

def evaluate_all_models_and_concepts_spearman(df,exclude_domain_agnostic=True):
    concepts = df['concept'].unique().tolist() + [None]
    results = []
    for concept in concepts:
        concept_result = evaluate_all_models_spearman(df,concept=concept,exclude_domain_agnostic=exclude_domain_agnostic)
        results.append(concept_result)
    return pd.concat(results)

def compare_model_pair_spearman(df,model1,model2,concept=None,exclude_domain_agnostic=True):
    n1, r1, p1 = evaluate_model_spearman(df,model1,concept,exclude_domain_agnostic)
    n2, r2, p2 = evaluate_model_spearman(df,model2,concept,exclude_domain_agnostic)
    # Fisher r-to-z transformation
    z1 = 0.5 * (np.log(1 + r1) - np.log(1 - r1))
    z2 = 0.5 * (np.log(1 + r2) - np.log(1 - r2))
    # Calculate standard error
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    # Calculate z-statistic
    z = (z1 - z2) / se
    # Calculate p-value
    p = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
    return r1,r2,z,p

def compare_all_models_spearman(df,concept=None,exclude_domain_agnostic=True):
    models = df['model'].unique()
    model_pairs = list(combinations(models,2))
    results = []
    for model_pair in model_pairs:
        model1,model2 = sorted(model_pair)
        r1, r2, z, p = compare_model_pair_spearman(df,model1,model2,concept=concept,exclude_domain_agnostic=exclude_domain_agnostic)
        concept_name = concept if concept else 'all_documents'
        results.append({'model1':model1,'model2':model2,'concept':concept_name,'r1':r1,'r2':r2,'z':z,'p':p})
    results = pd.DataFrame(results)
    return results


def compare_all_models_and_concepts_spearman(df,exclude_domain_agnostic=True):
    concepts = df['concept'].unique().tolist() + [None]
    results = []
    for concept in concepts:
        concept_result = compare_all_models_spearman(df,concept=concept,exclude_domain_agnostic=exclude_domain_agnostic)
        results.append(concept_result)
    results = pd.concat(results)
    return results


def evaluate_model_auc(df,model_name,concept=None,percent_thresh=0.3,exclude_domain_agnostic=True):
    df_sub = df[df['model']==model_name]
    if concept:
        df_sub = df_sub[df_sub['concept']==concept]
    if exclude_domain_agnostic:
        df_sub = df_sub[df_sub['concept'] != 'domain-agnostic']
    if len(df_sub) == 0:
        return
    
    y_true = df_sub['y_true']
    y_true = y_true >= percent_thresh
    y_pred = df_sub['y_pred']
    roc_auc = metrics.roc_auc_score(y_true,y_pred)
    return y_true,y_pred,roc_auc

def evaluate_all_models_auc(df,concept=None,percent_thresh=0.3,exclude_domain_agnostic=True):
    models = df['model'].unique()
    results = []
    for model in models:
        try:
            y_true,y_pred,roc_auc = evaluate_model_auc(df,model,concept=concept,percent_thresh=percent_thresh,exclude_domain_agnostic=exclude_domain_agnostic)
            concept_name = concept if concept else 'all_documents'
            results.append({'model':model,'concept':concept_name,'roc_auc':roc_auc})
        except:
            continue
    results = pd.DataFrame(results)
    return results

def evaluate_auc_all_models_and_concepts(df,percent_thresh=0.3,exclude_domain_agnostic=True):
    concepts = list(df['concept'].unique()) + [None]
    results = []
    for concept in concepts:
        concept_result = evaluate_all_models_auc(df,concept=concept,percent_thresh=percent_thresh,exclude_domain_agnostic=exclude_domain_agnostic)
        results.append(concept_result)
    results = pd.concat(results)
    return results


def evaluate_auc_different_thresholds(df,concept=None,exclude_domain_agnostic=True):
    thresholds = [0.1*x for x in range(1,10)]
    results = []
    for threshold in thresholds:
        threshold_results = evaluate_all_models_auc(df,concept=concept,percent_thresh=threshold,exclude_domain_agnostic=exclude_domain_agnostic)
        threshold_results['threshold'] = threshold
        results.append(threshold_results)
    results = pd.concat(results)
    return results


def auc_diff(y_true,y_pred_1,y_pred_2):
    return metrics.roc_auc_score(y_true,y_pred_1) - metrics.roc_auc_score(y_true,y_pred_2)

def compare_model_pair_auc(df,model1,model2,concept=None,percent_thresh=0.3,n_boots=100, exclude_domain_agnostic=True):
    y_true_1, y_pred_1, auc1 = evaluate_model_auc(df,model1,concept=concept,percent_thresh=percent_thresh,exclude_domain_agnostic=exclude_domain_agnostic)
    y_true_2, y_pred_2, auc2 = evaluate_model_auc(df,model2,concept=concept,percent_thresh=percent_thresh,exclude_domain_agnostic=exclude_domain_agnostic)
    data = (y_true_1,y_pred_1,y_pred_2)
    boot_result = scipy.stats.bootstrap(data,auc_diff,method='percentile',n_resamples=n_boots,paired=True)
    ci = boot_result.confidence_interval
    sig = False
    if ci[0] > 0 or ci[1] < 0:
        sig = True
    return auc1,auc2,sig


def compare_auc_all_models(df,concept=None,percent_thresh=0.3,exclude_domain_agnostic=True):
    results = []
    models = df['model'].unique().tolist()
    model_pairs = list(combinations(models,2))
    for model_pair in model_pairs:
        model1,model2 = model_pair
        try:
            auc1,auc2,sig = compare_model_pair_auc(df,model1,model2,concept=concept,percent_thresh=percent_thresh,exclude_domain_agnostic=exclude_domain_agnostic) #Evaluation over all documents
            print(model1,model2,auc1,auc2,sig)
            res = {'model1':model1,'model2':model2,'auc1':auc1,'auc2':auc2,'sig':sig}
            results.append(res)
        except:
            continue
    results = pd.DataFrame(results)
    return results

def run_all_spearman_evaluations(df, eval_dir):
    results_by_model_and_concept = evaluate_all_models_and_concepts_spearman(df, exclude_domain_agnostic=True)
    results_by_model_and_concept.to_csv(os.path.join(eval_dir, 'spearman_results_by_model_and_concept.tsv'), sep='\t', index=False)

    results_domain_agnostic_by_model = evaluate_all_models_spearman(df, concept='domain-agnostic', exclude_domain_agnostic=False)
    results_domain_agnostic_by_model.to_csv(os.path.join(eval_dir, 'spearman_results_domain_agnostic_by_model.tsv'), sep='\t', index=False)

    comparisons_by_model_and_concept = compare_all_models_and_concepts_spearman(df, exclude_domain_agnostic=True)
    comparisons_by_model_and_concept.to_csv(os.path.join(eval_dir, 'comparison_spearman_by_model_and_concept.tsv'), sep='\t', index=False)

    comparisons_domain_agnostic = compare_all_models_spearman(df, concept='domain-agnostic', exclude_domain_agnostic=False)
    comparisons_domain_agnostic.to_csv(os.path.join(eval_dir, 'comparison_spearman_domain_agnostic.tsv'), sep='\t', index=False)


def run_all_auc_evaluations_at_threshold(df, eval_dir, threshold=AUC_DEFAULT_THRESHOLD):
    results_by_model_and_concept = evaluate_auc_all_models_and_concepts(df, exclude_domain_agnostic=True)
    results_by_model_and_concept.to_csv(os.path.join(eval_dir, f'auc_results_by_model_and_concept_{threshold}.tsv'), sep='\t', index=False)

    results_domain_agnostic_by_model = evaluate_all_models_auc(df, concept='domain-agnostic', exclude_domain_agnostic=False)
    results_domain_agnostic_by_model.to_csv(os.path.join(eval_dir, f'auc_results_domain_agnostic_by_model_{threshold}.tsv'), sep='\t', index=False)

    comparisons_by_model_and_concept = compare_auc_all_models(df, exclude_domain_agnostic=True, percent_thresh=threshold)
    comparisons_by_model_and_concept.to_csv(os.path.join(eval_dir, f'comparison_auc_by_model_and_concept_{threshold}.tsv'), sep='\t', index=False)

    comparisons_domain_agnostic = compare_auc_all_models(df, concept='domain-agnostic', exclude_domain_agnostic=False, percent_thresh=threshold)
    comparisons_domain_agnostic.to_csv(os.path.join(eval_dir, f'comparison_auc_domain_agnostic_{threshold}.tsv'), sep='\t', index=False)


def run_all_auc_evaluations_at_different_thresholds(df, eval_dir, thresholds = [0.1*x for x in range(1,10)]):
    for percent_thresh in thresholds:
        run_all_auc_evaluations_at_threshold(df, eval_dir, threshold=percent_thresh)


def main():
    metaphor_scores_dir = METAPHOR_SCORES_DIR + '_for_eval'
    eval_dir = EVALUATION_DIR


    # Load the original data and word counts
    df_annot = load_ground_truth_annotations(ANNOTATED_DATA_FILE, LABEL_COL)
    # Load scores for ALL models in the metaphor scores eval directory. 
    models = [d for d in os.listdir(metaphor_scores_dir) if os.path.isdir(os.path.join(metaphor_scores_dir, d))]
    df_scores = load_all_model_scores(metaphor_scores_dir, CONCEPTS, models, mode = 'score',filename='scores_no_text.tsv')
    # Combine the annotations and scores
    df = combine_annotations_and_model_scores(df_annot, df_scores)

    # Run Spearman evaluations and comparisons
    run_all_spearman_evaluations(df, eval_dir)

    # Run AUC evaluations and comparisons at the default threshold (provided in config.py)
    run_all_auc_evaluations_at_threshold(df, eval_dir, threshold=AUC_DEFAULT_THRESHOLD)

    # Run AUC evaluations and comparisons at different thresholds
    run_all_auc_evaluations_at_different_thresholds(df, eval_dir)



   

  

   



if __name__ == '__main__':
    main()