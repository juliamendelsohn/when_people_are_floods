import json
import pandas as pd
import os
from math import log
from pathlib import Path
import re
import argparse
from config import FULL_DATASET_FILE, ANNOTATED_DATA_FILE, METAPHOR_SCORES_DIR, LLM_OUTPUT_DIR, CONCEPTS

"""
This script combines SBERT and LLM scores to create a full datasheet with scores.
It loads the original data, SBERT scores, and LLM scores, and combines them into a single dataframe.
The final scores are saved to a TSV file.
"""


def load_gpt_outputs(raw_gpt_output_file):
    with open(raw_gpt_output_file, 'r') as f:
        lines = f.readlines()
    outputs = []
    for line in lines:
        json_object= json.loads(line.strip())
        id_str = json_object['custom_id']
        output = json_object['response']['body']['choices'][0]['message']['content']
        output = output.replace('\n','').replace('\t','').replace('\r','').strip()
        try:
            output = json.loads(output)
            output = {k:v.lower() for k,v in output.items() if 'json' not in k.lower()}
        except:
            output = {}
        outputs.append({'id_str':id_str,'output':output})
    return pd.DataFrame(outputs)


def load_original_data(original_data_file):
    df_original = pd.read_csv(original_data_file,sep='\t',dtype=str)
    df_original['num_words'] = df_original['text'].apply(lambda x: len(x.split()))
    df_word_counts = df_original[['id_str','num_words']]
    return df_original, df_word_counts

def load_sbert_scores(sbert_scores_file):
    df_sbert = pd.read_csv(sbert_scores_file,sep='\t',dtype={'id_str':str})
    df_sbert.columns = ['sbert_'+col if col != 'id_str' else col for col in df_sbert.columns]
    return df_sbert

def load_llm_scores(llm_output_dir,model,strategy,concepts,df_word_counts):
    df_llm = load_gpt_outputs(os.path.join(llm_output_dir,f'{model}_{strategy}.jsonl'))
    df_llm = df_llm.merge(df_word_counts,on='id_str')
    for concept in concepts: 
        if concept == 'domain-agnostic':
            df_llm['count_'+concept] =  df_llm['output'].apply(lambda x: len(x))
        else:
            df_llm['count_'+concept] = df_llm['output'].apply(lambda x: sum([1 for item in x.items() if concept in item[1]]))
        df_llm['llm_'+concept] = df_llm.apply(lambda x: x['count_' + concept] / log(x['num_words'] + 1), axis=1)
    # remove all columns starting with count_
    cols = [col for col in df_llm.columns if not col.startswith('count_')]
    df_llm = df_llm[cols]
    return df_llm

def combine_sbert_and_llm_scores(df_original,df_sbert,df_llm,concepts):
    df_score = df_llm.merge(df_sbert,on='id_str')
    # for each concept, add the sbert and llm scores
    for concept in concepts:
        if concept != 'domain-agnostic':
            df_score['score_'+concept] = df_score['sbert_'+concept] + df_score['llm_'+concept]
        else:
            df_score['score_'+concept] = df_score['llm_'+concept]
    df_original = df_original.merge(df_score,on='id_str')
    return df_original



def main():
    parser = argparse.ArgumentParser(description='Create full datasheet with scores.')
    parser.add_argument('--embedding_model', type=str, default='all-miniLM-L6-v2',
                        help='The SBERT model used for embeddings (default: all-miniLM-L6-v2)')
    parser.add_argument('--llm', type=str, default='gpt-4o-2024-08-06',
                        help='The generative LLM used (default: gpt-4o-2024-08-06)')
    parser.add_argument('--prompt', type=str, default='zeroshot_description',
                        help='Name of the prompt used (default: zeroshot_description)')
    parser.add_argument('--eval', action='store_true',
                        help='If set, will load annotated data instead of full dataset')
    
    args = parser.parse_args()
    embedding_model_name = args.embedding_model
    llm = args.llm
    prompt = args.prompt
    eval_mode = args.eval
    if eval_mode:
        original_data_file = ANNOTATED_DATA_FILE
        metaphor_scores_dir = METAPHOR_SCORES_DIR + '_for_eval'
        llm_output_dir = LLM_OUTPUT_DIR + '_for_eval'
    else:
        original_data_file = FULL_DATASET_FILE
        metaphor_scores_dir = METAPHOR_SCORES_DIR
        llm_output_dir = LLM_OUTPUT_DIR

    Path(metaphor_scores_dir).mkdir(parents=True, exist_ok=True)
    Path(llm_output_dir).mkdir(parents=True, exist_ok=True)

    sbert_scores_file = os.path.join(metaphor_scores_dir, f'sbert_scores_{embedding_model_name}.tsv')
    scores_out_dir = os.path.join(metaphor_scores_dir, f"{embedding_model_name}_{llm}_{prompt}")
    Path(scores_out_dir).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(scores_out_dir, f'scores_full.tsv')
    out_file_no_text = os.path.join(scores_out_dir, f'scores_no_text.tsv')

    df_original, df_word_counts = load_original_data(original_data_file)
    df_sbert = load_sbert_scores(sbert_scores_file)
    df_llm = load_llm_scores(llm_output_dir, llm, prompt, CONCEPTS, df_word_counts)
    df = combine_sbert_and_llm_scores(df_original, df_sbert, df_llm, CONCEPTS)
    df_no_text = df.drop(columns=['text'])

    print(df)
    print(df.columns)
    
    df.to_csv(out_file, sep='\t', index=False)
    df_no_text.to_csv(out_file_no_text,sep='\t',index=False)



if __name__ == '__main__':
    main()