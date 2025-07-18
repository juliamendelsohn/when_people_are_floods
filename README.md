##  When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models

This repository contains materials for the ACL 2025 paper _"When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models"_ by Julia Mendelsohn and Ceren Budak. 

The paper is available on [arXiv here](https://arxiv.org/abs/2502.13246).

Our main contributions are:
1. A novel computational approach that combines word-level and document-level signals to measure metaphorical language with respect to specified source domain concepts.
2. An analysis of dehumanizing metaphors in immigration discourse on social media, focused on the relationship between metaphor, political ideology, and user engagement. 

As per X's Terms of Service, we can only publicly share very small samples of full-text data. 
The complete datasets for annotation/evaluation and analysis are available upon request. Please contact us at `juliame@umd.edu` and `cbudak@umich.edu`.

## Repository Structure

This repository includes three folders: 
- `data/`: Contains dataset samples and texts required to run the models, 
    -   `annotated_data_sample_with_text.tsv`: A small sample of the annotated data used for evaluation, including text.
    -   `annotated_data_id_only.tsv`: The full annotated dataset used for evaluation, with only post IDs (no text).
    -   `sample_tweet_data_with_text.tsv`: A small sample of our dataset for analysis.
    -   `carrier_sentences.tsv`: File containing "carrier sentences" needed for the discourse-level (document embedding) measurement.
    -   `prompts/`: Folder containing .txt files with prompts used for the word-level (LLM) measurement.
- `code/`: Contains code used for the paper, separated into subfolders for each study component.
    -  `0_annotation/`: Folder containing code to create the annotation sample, process Qualtrics output files, and calculate agreement. Probably not useful for most users.
    -  `1_models/`: Folder containing code to run the word-level (LLM) and document-level (discourse embedding) models.
        - `0_run_sbert.py`: Code to run the sentence-BERT model for document embeddings.
        - `1a_submit_gpt_batch.py`: Code to prepare and submit batches of prompts to OpenAI for GPT-based word-level measurements.
        - `1b_save_gpt_output.py`: Code to process the output from OpenAI
        - `2_run_llama_huggingface.py`: Code to run and save the Llama-3 model for word-level measurements.
    -  `2_scoring_and_evaluation/`: Aggregate metaphor scores and evaluate model performance.
        - `0_calculate_metaphor_scores.py`: Aggregate metaphor scores from the word-level and document-level models.
        - `1_evaluate_and_compare_models.py`: Evaluate and compare model performance on annotated data.
    -  `3_analysis/`: Code to analyze the results of the models. (in R, with bash scripts)
        - `regression_ideology_metaphor.py`: Analyze the relationship between metaphor usage and political ideology.
        - `regression_metaphor_engagement.py`: Analyze the relationship between metaphor usage and user engagement, and the moderating effect of political ideology.
    -  `4_notebooks/`: Miscellaneous Jupyter notebooks for exploratory analysis and visualization.
- `results/`: Contains lots of evaluation and analysis results in subfolders. 
    - Also includes the full datasheet for the regression analyses: `tweet_data_with_metaphor_scores.tsv.gz`. 


## Citation
If you use this code or data in your research, please cite our paper as follows:

Mendelsohn, J., & Budak, C. (2025). When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models. arXiv preprint arXiv:2502.13246.

```bibtex 
@article{mendelsohn2025people,
  title={When People are Floods: Analyzing Dehumanizing Metaphors in Immigration Discourse with Large Language Models},
  author={Mendelsohn, Julia and Budak, Ceren},
  journal={arXiv preprint arXiv:2502.13246},
  year={2025}
}
``` 