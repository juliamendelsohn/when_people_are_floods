import os

#Set project directory to one directory above this file
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PROMPT_DIR = os.path.join(DATA_DIR, 'prompts') # Each prompt file should be named [strategy].txt (e.g. zeroshot_description.txt)

RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')



LLM_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'llm_outputs')
METAPHOR_SCORES_DIR = os.path.join(RESULTS_DIR, 'metaphor_scores')
EVALUATION_DIR = os.path.join(RESULTS_DIR, 'evaluation')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')



# All tweets that you want to score with the LLMs. Requires at minimum, 'text' and 'id_str' columns.
FULL_DATASET_FILE = os.path.join(DATA_DIR, 'sample_tweet_data_with_text.tsv') 

# Annotated dataset. Requires at minimum, 'text, 'id_str', and a column with ground truth labels.

ANNOTATION_DIR = os.path.join(DATA_DIR,'annotations')
ANNOTATED_DATA_FILE = os.path.join(ANNOTATION_DIR, 'annotated_data_sample_with_text.tsv')
LABEL_COL = 'percent_yes' # Column with ground truth labels in annotated data





CONCEPTS = ['animal','commodity','parasite','pressure','vermin','war','water','domain-agnostic']

CARRIER_SENTENCES_FILE = os.path.join(DATA_DIR, 'carrier_sentences.tsv')



# FULL_DATASHEET_FILE = '/nfs/turbo/si-juliame/metaphor/results/full_datasheets/2024-05-20/filtered_tweet_data_shuffled_results_with_scores.tsv'

OPENAI_BATCH_SIZE = 1000 # Number of tasks per batch for OpenAI API (can be adjusted based on API limits) 
AUC_DEFAULT_THRESHOLD = 0.3 # Default threshold for AUC evaluation