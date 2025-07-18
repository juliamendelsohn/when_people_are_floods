import simpledorff
import pandas as pd 
import os
from config import ANNOTATION_DIR, ALL_ANNOTATIONS_FILE


"""
This script calculates Krippendorff's alpha for the annotations. 
Agreement is calculated for the full set of annotations, as well as for subsets based on source domain and annotator ideology.
The results are saved in a file named 'agreement.tsv' in the annotations directory.
"""


def get_alpha(df):
    # Fix the division by zero error
    alpha = simpledorff.calculate_krippendorffs_alpha_for_df(df,
                experiment_col='id_str',annotator_col='PROLIFIC_PID',class_col='label')
    return alpha


def get_alpha_on_subset(df,concept=None,ideology=None):
    df_sub = df.copy()
    if concept is not None:
        df_sub = df_sub[df_sub['concept'] == concept]
    if ideology is not None:
        df_sub = df_sub[df_sub['Ideology'] == ideology]
    alpha = get_alpha(df_sub)
    return alpha
   


def get_all_alpha(df):
    results = []
    full_result = ['all_documents','all_annotators',get_alpha_on_subset(df)]
    results.append(full_result)
    for concept in df['concept'].unique():
        res = [concept,'all_annotators',get_alpha_on_subset(df,concept=concept)]
        results.append(res)
    for ideology in df['Ideology'].unique():
        res = ['all_documents',ideology,get_alpha_on_subset(df,ideology=ideology)]
        results.append(res)
    for concept in df['concept'].unique():
        for ideology in df['Ideology'].unique():
            res = [concept,ideology,get_alpha_on_subset(df,concept=concept,ideology=ideology)]
            results.append(res)
    results = pd.DataFrame(results,columns=['concept','ideology','alpha'])
    return results




def main():
    out_file = os.path.join(ANNOTATION_DIR,'agreement.tsv')
    df = pd.read_csv(ALL_ANNOTATIONS_FILE,sep='\t')
    results = get_all_alpha(df)
    results.to_csv(out_file,sep='\t',index=False)
    print(results)


if __name__ == '__main__':
    main()