import os
from pathlib import Path
import pandas as pd
from huggingface_hub import InferenceClient
import time
import argparse
from config import FULL_DATASET_FILE, ANNOTATED_DATA_FILE, PROMPT_DIR, LLM_OUTPUT_DIR

"""
This script runs a batch of tasks using the Hugging Face Inference API.
The model and prompt strategy are specified as command line arguments (--model and --strategy).
The model is a specified Hugging Face model (e.g., meta-llama/Meta-Llama-3.1-70B-Instruct).
The prompt is read from a file in the prompts dir named by the specified strategy (e.g., zeroshot_description.txt).
Note that a Hugging Face API token is required to run this script, and should be stored in the environment variable HF_API_TOKEN.
"""


def read_prompt_from_file(prompt_file):
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt

def prepare_messages(input_file,prompt):
    df = pd.read_csv(input_file, sep='\t',dtype=str)
    df = df.drop_duplicates(subset='id_str')
    messages = []
    id_list = []
    for index, row in df.iterrows():
        tweet = row['text'].strip()
        message = prompt + '\nTweet: ' + tweet
        messages.append(message)
        id_list.append(row['id_str'])

    return messages, id_list


def get_llm_output(client,messages,id_list,model,output_file):
    with open(output_file, 'w') as f:
        f.write('id_str\toutput\n')
        for message,id_str in zip(messages,id_list):
            completions = client.chat_completion(
                model=model,
                messages=[{"role":"user","content":message}],
                temperature =  0,
                stream = False)
            # Add id_str to response
            output = completions.choices[0].message.content.replace('\n',' ').replace('\t',' ').replace('\r',' ')
            f.write(id_str + '\t' + output)
            f.write('\n')
            # Sleep for 5 seconds to avoid rate limiting
            time.sleep(5)
            



def main():

    parser = argparse.ArgumentParser(description='Process tasks with Hugging Face API.')
    parser.add_argument('--model', type=str, required=True, help='Hugging Face model to use (e.g., meta-llama/Meta-Llama-3.1-70B-Instruct)')
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
    model_name_for_file = model.split('/')[-1]  # Get the model name from the full path
    output_file = os.path.join(llm_output_dir,f'{model_name_for_file}_{strategy}.jsonl')


    prompt_dir = PROMPT_DIR
    prompt_file = os.path.join(prompt_dir,strategy+'.txt')
    prompt = read_prompt_from_file(prompt_file)

    client = InferenceClient(
        model,
        token=os.environ['HF_API_TOKEN'],
    )

    messages,id_list = prepare_messages(input_file,prompt)
    get_llm_output(client,messages,id_list,model,output_file)

   

    
if __name__ == '__main__':
    main()