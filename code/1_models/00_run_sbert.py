import pandas as pd
import csv
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score
import torch
import numpy as np
import argparse
from pathlib import Path
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from config import *


def load_concept_sentences(concept_filename):
    df_concept = pd.read_csv(concept_filename, sep='\t')
    concept_sentences = {}
    for index, row in df_concept.iterrows():
        concept = row['concept']
        sentence = row['sentence']
        if concept not in concept_sentences:
            concept_sentences[concept] = []
        concept_sentences[concept].append(sentence)
    return concept_sentences

def get_concept_embeddings(model,concept_sentences):
    concept_embeddings = {}
    for concept in concept_sentences:
        sentences = concept_sentences[concept]
        sent_embeds = model.encode(sentences,normalize_embeddings=True)
        concept_embeddings[concept] = np.mean(sent_embeds,axis=0)
    df = pd.DataFrame(concept_embeddings).transpose()
    return df
    
def load_text_data(tweet_filename):
    df = pd.read_csv(tweet_filename, sep='\t', dtype=str)
    return df[['text','id_str']]

def get_concept_associations(model,tweets,concept_embeddings):
    tweet_embeddings = model.encode(tweets['text'],normalize_embeddings=True)
    scores = dot_score(tweet_embeddings,torch.tensor(concept_embeddings.values))
    df = pd.DataFrame(scores.numpy(),columns=concept_embeddings.index)
    df['id_str'] = tweets['id_str']
    print(df)
    return df

def save_concept_associations(scores,out_filename):
    scores.to_csv(out_filename,sep='\t',index=False)


def main():

    parser = argparse.ArgumentParser(description='Run SBERT to compute concept associations.')
    parser.add_argument('--embedding_model', type=str, default='all-miniLM-L6-v2', 
                        help='The SBERT model to use for embeddings.')
    parser.add_argument('--eval', action='store_true',
                        help='If set, will load annotated data instead of full dataset')
    args = parser.parse_args()
    embedding_model_name = args.embedding_model
    eval_mode = args.eval
    if eval_mode:
        dataset_file = ANNOTATED_DATA_FILE
        metaphor_scores_dir = f"{METAPHOR_SCORES_DIR}_for_eval"
    else:
        dataset_file = FULL_DATASET_FILE
        metaphor_scores_dir = METAPHOR_SCORES_DIR  
    Path(metaphor_scores_dir).mkdir(parents=True, exist_ok=True) 

    model = SentenceTransformer(embedding_model_name)

    carrier_sentences = load_concept_sentences(CARRIER_SENTENCES_FILE)
    concept_embeddings = get_concept_embeddings(model,carrier_sentences)
    texts = load_text_data(dataset_file)
    scores = get_concept_associations(model,texts,concept_embeddings)

    sbert_score_outfile = os.path.join(metaphor_scores_dir, f'sbert_scores_{embedding_model_name}.tsv')
    save_concept_associations(scores,sbert_score_outfile)



if __name__ == '__main__':
    main()