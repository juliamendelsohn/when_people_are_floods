import json
from openai import OpenAI
import pandas as pd
import os
from pathlib import Path
import time
import argparse
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from config import FULL_DATASET_FILE, ANNOTATED_DATA_FILE, PROMPT_DIR, LLM_OUTPUT_DIR, OPENAI_BATCH_SIZE

"""
This script prepares tasks for the OpenAI Batch API and uploads them for processing.
The model is a specified OpenAI model (e.g., gpt-4o-2024-08-06).
The prompt is read from a file in the prompts dir named by the specified strategy (e.g., zeroshot_description.txt).
The model and prompt strategy are specified as command line arguments (--model and --strategy).
"""

client = OpenAI()

def read_prompt_from_file(prompt_file):
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt


def create_task_args(row,model,prompt):
    tweet = row['text'].strip()
    task = {
        "custom_id": row['id_str'],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0,
            "response_format": { 
                "type": "json_object"
            },
            "messages": [
                {
                    "role": "user",
                    "content": prompt + '\nTweet: ' + tweet
                }
            ],
        }
    }
    return task


def write_tasks_to_file(tasks, output_file):
    with open(output_file, 'w') as f:
        for obj in tasks:
            f.write(json.dumps(obj) + '\n')


def prepare_tasks(input_file,output_prefix,prompt,model,start_idx,end_idx):
    df = pd.read_csv(input_file, sep='\t',dtype=str)
    df = df.drop_duplicates(subset='id_str')
    tasks = []
    for index, row in df.iterrows():
        if index >= start_idx and index < end_idx:
            task = create_task_args(row,model,prompt)
            tasks.append(task)
    print(f'Number of tasks: {len(tasks)}')
    output_file = f'{output_prefix}_start{start_idx}_end{end_idx}.jsonl'
    write_tasks_to_file(tasks, output_file)
    return output_file



def upload_file(batch_filename):
    batch_file = client.files.create(
        file=open(batch_filename, "rb"),
        purpose="batch"
    )
    return batch_file


def create_batch_job(batch_file):
    batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
    )
    return batch_job




def main():

    parser = argparse.ArgumentParser(description='Batch process tasks with OpenAI API.')
    parser.add_argument('--model', type=str, required=True, help='The OpenAI model to use (e.g., gpt-4o-2024-08-06)')
    parser.add_argument('--strategy', type=str, required=True, help='The strategy for the prompt (e.g., zeroshot_description)')
    parser.add_argument('--eval', action='store_true',
                        help='If set, will load annotated data instead of full dataset')
    args = parser.parse_args()

    model = args.model
    strategy = args.strategy
    eval_mode = args.eval

    if eval_mode:
        input_file = ANNOTATED_DATA_FILE
        llm_output_dir = LLM_OUTPUT_DIR + '_for_eval'
    else:
        input_file = FULL_DATASET_FILE
        llm_output_dir = LLM_OUTPUT_DIR

    Path(llm_output_dir).mkdir(parents=True, exist_ok=True)
    batch_task_prefix = os.path.join(llm_output_dir, f'task_batch_{model}_{strategy}') 

    prompt_dir = PROMPT_DIR
    prompt_file = os.path.join(prompt_dir, strategy + '.txt')
    prompt = read_prompt_from_file(prompt_file)

    num_tasks = len(pd.read_csv(input_file, sep='\t', dtype=str))
    batch_size = OPENAI_BATCH_SIZE
    idx = 0
    while idx < num_tasks: 
        batch_filename = prepare_tasks(input_file, batch_task_prefix, prompt, model, idx, idx + batch_size)
        print(batch_filename)
        batch_file = upload_file(batch_filename)
        batch_job = create_batch_job(batch_file)
        while batch_job.status != 'completed':
            batch_job = client.batches.retrieve(batch_job.id)
            print(idx,batch_job.id,batch_job.status)
            time.sleep(120)
        idx += batch_size





    




    




if __name__ == "__main__":
    main()
