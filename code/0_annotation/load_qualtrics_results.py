import pandas as pd 
import os
import json
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
from datetime import date
import statsmodels.stats.proportion as proportion
from itertools import combinations
from config import DATA_DIR, CONCEPTS

""" 
This script loads and reformats the results of the Qualtrics annotation tasks. 
It loads the demographics, annotations, and filters the responses based on time and response diversity.
It also computes statistics on the annotations, such as counts of metaphorical responses and proportions across ideologies.
The script is shared for transparency but probably isn't useful because we don't include the raw Qualtrics output files in the public repo.
"""



def load_demographics(demographics_file):
    df = pd.read_csv(demographics_file)
    df = df[df['Status'].isin(['APPROVED','AWAITING REVIEW'])]
    # Recode Immigration
    df['Immigrant'] = df['Immigration'].replace({
        'Yes, I was born in the country I am now living in':'No',
        'No, I moved to the country I am now living in':'Yes'})
    df = df.drop(columns=['Immigration'])
    df = df.rename(columns={'Political spectrum (us)':'Ideology','Participant id':'PROLIFIC_PID'})
    demo_cols = ['PROLIFIC_PID','Ideology', 'Immigrant', 'Age','Sex']
    df = df[demo_cols]

    return df

def load_qualtrics_annotations(annotation_file,sample_file,concept):
    sample_df = pd.read_csv(sample_file,sep='\t',dtype=str)
    id_list = sample_df['id_str'].tolist()
    df = pd.read_csv(annotation_file,sep='\t',encoding='utf-16')
    df = df.drop(index=[0,1]).reindex()
    annot_cols = [x for x in df.columns if  x.endswith(concept)]
    other_cols = [x for x in df.columns if x not in annot_cols]
    rename_map = {annot_cols[i]:id_list[i] for i in range(len(annot_cols))}
    id_list = id_list[:len(annot_cols)]
    df = df.rename(columns=rename_map)
    df = pd.melt(df,value_vars=id_list,id_vars=other_cols,var_name='id_str',value_name='label')
    df = df.dropna(subset=['label'])
    df = df.merge(sample_df,on='id_str')
    df = df[df['Status']=='IP Address']
    df = df[df['Finished'] == 'True']   
    cols_to_keep = ['PROLIFIC_PID','Duration (in seconds)','id_str','text','label']
    df = df[cols_to_keep]
    return df


def filter_bad_responses(df,min_time=0,require_response_diversity=True,convert_unsure_to=None,
                         annotation_count_threshold=5):
    time_col = 'Duration (in seconds)'
    # convert df[time_col] to numeric
    df[time_col] = pd.to_numeric(df[time_col],errors='coerce')
    df = df[df[time_col] >= min_time]
    if require_response_diversity:
        annotator_counts = df.groupby('PROLIFIC_PID')['label'].nunique()
        annotators_to_keep = annotator_counts[annotator_counts >= 2].index
        df = df[df['PROLIFIC_PID'].isin(annotators_to_keep)]
    if convert_unsure_to is not None:
        if convert_unsure_to == 'Remove':
            df = df[df['label'] != 'Unsure']
        if convert_unsure_to == 'No':
            df['label'] = df['label'].replace('Unsure','No')
        if convert_unsure_to == 'Yes':
            df['label'] = df['label'].replace('Unsure','Yes')
    id_counts = get_annotation_counts(df)
    id_counts = id_counts[id_counts['num_annotations'] >= annotation_count_threshold]
    df = df[df['id_str'].isin(id_counts['id_str'])]
    return df


def load_concept_and_demographics(qualtrics_dir,sample_dir,concept):
    file_pattern = f'qualtrics_{concept}'
    annotation_files = [x for x in os.listdir(qualtrics_dir) if x.startswith(file_pattern)]
    annotation_files = [os.path.join(qualtrics_dir,x) for x in annotation_files]
    sample_file = os.path.join(sample_dir,f'{concept}.tsv')
    dfs = []
    for annotation_file in annotation_files:
        df_file = load_qualtrics_annotations(annotation_file,sample_file,concept)
        dfs.append(df_file)
    df = pd.concat(dfs)

    demo = load_demographics(os.path.join(qualtrics_dir,f'demographics_{concept}.csv'))
    df = df.merge(demo,on='PROLIFIC_PID')
    return df



def get_annotation_counts(df):
    sample_df = df[['id_str','text']].drop_duplicates()
    id_counts = df['id_str'].value_counts()
    id_counts = id_counts.reset_index()
    id_counts.columns = ['id_str','num_annotations']
    id_counts = sample_df.merge(id_counts,on='id_str')
    id_counts = id_counts.sort_values('num_annotations',ascending=True)
    return id_counts

    
def get_annotation_count_stats(df,counts_dir):
    counts = []
    for concept in df['concept'].unique():
        concept_df = df[df['concept'] == concept]
        unique_docs = concept_df['id_str'].nunique()
        counts.append((concept,unique_docs,len(concept_df)))
    counts = pd.DataFrame(counts,columns=['concept','documents','annotations'])
    outfile = os.path.join(counts_dir,'annotation_counts.tsv')
    counts.to_csv(outfile,sep='\t',index=False)

    id_counts = get_annotation_counts(df)
    # Make table to plot histogram of counts
    count_hist_table = id_counts['num_annotations'].value_counts().reset_index()
    count_hist_table.columns = ['num_annotations','num_documents']
    count_hist_table = count_hist_table.sort_values('num_annotations')
    outfile = os.path.join(counts_dir,'annotation_counts_by_doc.tsv')
    count_hist_table.to_csv(outfile,sep='\t',index=False)

def get_label_counts(df):
    yes_counts = len(df[df['label'] == 'Yes'])
    total_counts = len(df)
    percent_yes = yes_counts / total_counts
    return yes_counts,total_counts,percent_yes

def get_all_label_stats(df,counts_dir):
    results = []
    total_result = ['all_documents','all_annotators'] + list(get_label_counts(df))
    results.append(total_result)

    for ideology in df['Ideology'].unique():
        ideology_df = df[df['Ideology'] == ideology]
        res = ['all_documents',ideology] + list(get_label_counts(ideology_df))
        results.append(res)
    
    for concept in df['concept'].unique():
        concept_df  = df[df['concept'] == concept]
        res = [concept,'all_annotators'] + list(get_label_counts(concept_df))
        results.append(res)
    
    for concept in df['concept'].unique():
        for ideology in df['Ideology'].unique():
            sub_df = df[(df['concept'] == concept) & (df['Ideology'] == ideology)]
            res = [concept,ideology] + list(get_label_counts(sub_df))
            results.append(res)
        
    results = pd.DataFrame(results,
            columns=['concept','ideology','num_metaphorical','num_annotations','percent_metaphorical'])
    print(results)
    outfile = os.path.join(counts_dir,'label_counts.tsv')
    results.to_csv(outfile,sep='\t',index=False)
    return results

def test_proportion_diff(results,counts_dir):
    ideologies = ['Liberal','Moderate','Conservative']
    stats = []
    for concept in results['concept'].unique():
        df = results[results['concept']==concept]
        pairs = list(combinations(ideologies,2))
        for pair in pairs:
            ideology1,ideology2 = sorted(pair)
            ideology1_count = df[df['ideology'] == ideology1]['num_metaphorical'].values[0]
            ideology2_count = df[df['ideology'] == ideology2]['num_metaphorical'].values[0]
            ideology1_nob = df[df['ideology'] == ideology1]['num_annotations'].values[0]
            ideology2_nob= df[df['ideology'] == ideology2]['num_annotations'].values[0]
            count = [ideology1_count,ideology2_count]
            nobs = [ideology1_nob,ideology2_nob]
            ztest,p = proportion.proportions_ztest(count, nobs, value=None)
            stat_res = [concept,ideology1,ideology2,ztest,p]
            stats.append(stat_res)
    stats = pd.DataFrame(stats,columns=['concept','ideology1','ideology2','ztest','p'])
    outfile = os.path.join(counts_dir,'label_proportion_tests_across_ideology.tsv')
    stats.to_csv(outfile,sep='\t',index=False)


    




def get_label_distribution_per_document(df):
    # Get distributions of labels per document
    doc_counts = df.groupby(by=['id_str','concept'])['label'].value_counts()
    # Get percentage of "no" and yes responses per document
    doc_counts = doc_counts.unstack().fillna(0)
    doc_counts['total'] = doc_counts.sum(axis=1)
    doc_counts['percent_no'] = doc_counts['No'] / doc_counts['total']
    doc_counts['percent_yes'] = doc_counts['Yes'] / doc_counts['total']
    doc_counts = doc_counts.reset_index()
    doc_counts.columns = ['id_str'] + list(doc_counts.columns[1:])
    return doc_counts



def main():
    date = '2024-10-07'
    base_dir = DATA_DIR
    sample_dir = os.path.join(base_dir,f'samples_for_annotation_{date}')
    qualtrics_dir = os.path.join(base_dir,f'qualtrics_output_{date}')
    annotations_dir = os.path.join(base_dir,f'annotations_{date}')
    counts_dir = os.path.join(base_dir,f'counts_annotation_{date}')
    if not os.path.exists(annotations_dir):
        os.mkdir(annotations_dir)
    if not os.path.exists(counts_dir):
        os.mkdir(counts_dir)

    concepts = CONCEPTS
    all_dfs = []    
    for concept in concepts:
        df = load_concept_and_demographics(qualtrics_dir,sample_dir,concept)
        df = filter_bad_responses(df,
                                  min_time=180,
                                  require_response_diversity=True,
                                  convert_unsure_to='Remove', # yes, no, remove, None
                                  annotation_count_threshold=0
                                  )
        df['concept'] = concept
        all_dfs.append(df)
    
    df = pd.concat(all_dfs)
    df.to_csv(os.path.join(annotations_dir,'all_annotations.tsv'),sep='\t',index=False)
    get_annotation_count_stats(df,counts_dir)
    results = get_all_label_stats(df,counts_dir)
    test_proportion_diff(results,counts_dir)
    df_by_doc = get_label_distribution_per_document(df)
    df_by_doc.to_csv(os.path.join(annotations_dir,'all_annotations_percent_metaphorical.tsv'),sep='\t',index=False)
    


    

    

if __name__ == '__main__':
    main()
