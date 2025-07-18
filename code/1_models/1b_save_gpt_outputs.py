from openai import OpenAI
import pandas as pd
import os
import json
from pathlib import Path
import argparse
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from config import LLM_OUTPUT_DIR


def get_filename(client,batch_job):
    try:
        batch_file = client.files.retrieve(batch_job.input_file_id)
        input_file_name = batch_file.filename 
        return input_file_name
    except:
        return None

def get_gpt_output(client,batch_job):
    result_job_id = batch_job.output_file_id
    if result_job_id is None:
        return None
    result = client.files.content(result_job_id).content
    results = []
    for line in result.splitlines():
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)
    return results

def get_gpt_output_all_batches(client,prefix):
    batch_jobs = client.batches.list()
    all_results = []
    for i,batch_job in enumerate(batch_jobs):
        input_file_name = get_filename(client,batch_job)
        if input_file_name and input_file_name.startswith(prefix):
            results = get_gpt_output(client,batch_job)
            if results is not None:
                print(input_file_name)
                all_results += results
    return all_results

def save_gpt_output(output_file_name,results):
    with open(output_file_name, 'w') as f:
        for obj in results:
            f.write(json.dumps(obj) + '\n')

def save_parsed_gpt_output(result_dir,input_file_name,results):
    res_parsed = []
    for res in results:
        id_str = str(res['custom_id'])
        output = res['response']['body']['choices'][0]['message']['content']
        output = json.loads(output)
        print(id_str,output)
        res_parsed.append({'id_str':id_str,'output':output})
    df = pd.DataFrame(res_parsed)
    output_file_name = os.path.join(result_dir,input_file_name.replace('.jsonl','.tsv'))
    df.to_csv(output_file_name,sep='\t',index=False)
    print(df)
        
def save_all_batches(client, result_dir,prefix):
    raw_result_dir = Path(os.path.join(result_dir, 'raw'))
    parsed_result_dir = Path(os.path.join(result_dir, 'parsed'))
    raw_result_dir.mkdir(parents=True, exist_ok=True)
    parsed_result_dir.mkdir(parents=True, exist_ok=True)

    batch_jobs = client.batches.list()
    for batch_job in batch_jobs:
        input_file_name = get_filename(client,batch_job)
        if input_file_name.startswith(prefix):
            results = get_gpt_output(client,batch_job)
            if results is not None:
                save_gpt_output(raw_result_dir,input_file_name,results)
                save_parsed_gpt_output(parsed_result_dir,input_file_name,results)


def main():
    
    parser = argparse.ArgumentParser(description='Save outputs from OpenAI batches.')
    parser.add_argument('--model', type=str, required=True, default='gpt-4o-2024-08-06',
                        help='The OpenAI model used (e.g., gpt-4o-2024-08-06)')
    parser.add_argument('--strategy', type=str, required=True, default='zeroshot_description',
                        help='The strategy for the prompt (e.g., zeroshot_description)')
    parser.add_argument('--eval', action='store_true',
                        help='If set, will load outputs to an eval directory')
    args = parser.parse_args()
    model = args.model
    strategy = args.strategy
    eval_mode = args.eval

    if eval_mode:
        llm_output_dir = LLM_OUTPUT_DIR + '_for_eval'
    else:
        llm_output_dir = LLM_OUTPUT_DIR

    Path(llm_output_dir).mkdir(parents=True, exist_ok=True)
    output_file_name = os.path.join(llm_output_dir,f'{model}_{strategy}.jsonl')
    batch_prefix = f'task_batch_{model}_{strategy}'

    client = OpenAI()
    all_results = get_gpt_output_all_batches(client,batch_prefix)
    save_gpt_output(output_file_name,all_results)

    
    
if __name__ == "__main__":
    main()