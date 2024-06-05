# script to print the list of edited documents experiment as a latex table

import pandas as pd
import argparse 
import logging 
import numpy as np
import os
import re

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="report results of edited documents")
    parser.add_argument('-edited-results-file', type=str, help='file with results of edited documents', default="out.csv")
    parser.add_argument('-null-results-file', type=str, help='file with results of non-edited documents', default="out_null.csv") 

    args = parser.parse_args()

    results_edited_filename = args.edited_results_file
    results_null_filename = args.null_results_file

    df = pd.read_csv(results_edited_filename)
    # get basename of the file, and extract the title of the document.
    # column 0 contains the filename, and the title is the part between the last '/' and '_edited.txt'
    basenames = df.iloc[:,0].apply(os.path.basename)
    # now extract the name of the document
    df.loc[:, 'title'] = [re.findall(r"([A-Za-z \(\)]+)_edited[1-10]?\.*", x)[0] for x in basenames]
    #df.loc[:, 'title'] = df.iloc[:,0].str.extract(r"/([A-Za-z \(\)]+)_edited[1-10]?.txt")
    
    logging.info(f"Loaded {len(df)} rows from {results_edited_filename}")

    df0 = pd.read_csv(results_null_filename)
    
    # first extract the basename of the file:
    basenames = df0.iloc[:,0].apply(os.path.basename)
    # now extract the name of the document
    df0.loc[:, 'title'] = [re.findall(r"([A-Za-z \(\)]+)\.*", x)[0] for x in basenames]
    
    logging.info(f"Loaded {len(df0)} rows from {results_null_filename}")
    
    df['F1'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])
    df['TPR'] = df['recall']
    df_disp = df.merge(df0, on = 'title', suffixes=["", " (null)"], how='inner').filter(
        ['title', 'length', 'HC (null)', 'HC_pvalue (null)', 'edit_rate','HC', 'HC_pvalue', 'bonf', 'F1'])
    print("Could not find a match to the following titles:")
    print(df_disp[df_disp.isna().any(axis=1)]['title'])
    df_disp = df_disp.dropna()

    columns = ['title',  'edit_rate', 'length', 'HC', 'HC_pvalue', 'HC (null)',  'HC_pvalue (null)',  'bonf', 'F1']

    def round_and_bold_blue(x):
        return f"\\color{{blue}} \\textbf{{{np.round(x,3)}}}"
        
    def round_and_bold_red(x):
        return f"\\color{{red}} \\textbf{{{np.round(x,3)}}}"
        
    df_disp = df_disp.filter(columns)
    df_disp.loc[df_disp['HC_pvalue'] > 0.05, 'F1'] = "NA"
    df_disp.loc[:,'length'] = df_disp['length'].apply(lambda x: f"{int(x)}")
    df_disp.loc[:, 'title'] = df_disp['title'].apply(lambda x: f"\\texttt{{{x}}}")
    sig_pval = df_disp['HC_pvalue'] < 0.05
    df_disp.loc[sig_pval, 'HC'] = df_disp.loc[sig_pval, 'HC'].apply(round_and_bold_blue)
    sig_pval_null = df_disp['HC_pvalue (null)'] < 0.05
    df_disp.loc[sig_pval_null, 'HC (null)'] = df_disp.loc[sig_pval_null, 'HC (null)'].apply(round_and_bold_red)
    sig_pval_bonf = df_disp['bonf'] < 0.05
    df_disp.loc[sig_pval_bonf, 'bonf'] = df_disp.loc[sig_pval_bonf, 'bonf'].apply(round_and_bold_red)
    #df_disp.loc[:, 'HC_pvalue (null)'] = df_disp['HC_pvalue (null)'].apply(round_and_bold_red)
    #df_disp.loc[:, 'HC_pvalue'] = df_disp['HC_pvalue'].apply(round_and_bold)
    # for all column name, replace all '_' in column title with '-'
    df_disp.columns = df_disp.columns.str.replace("_", "-")
    aa = df_disp.set_index('title')
    print(aa.to_latex(float_format=lambda x: '%.2f' % x))


def merge_results_of_two_models(df_disp1, df_disp2, model1, model2):
    """
    Merge the results of two models into a single dataframe, and print the results as a latex table
    """
    
    df_merged = df_disp1.merge(df_disp2, on = ['title', 'edit_rate', 'length'], suffixes=[" (gpt2)", " (phi2)"], how='outer').filter(
    ['title', 'length', 'edit_rate', 'HC (null) (gpt2)', 'HC_pvalue (null) (gpt2)','HC (gpt2)', 'HC_pvalue (gpt2)', 'bonf (gpt2)', 
     'HC (null) (phi2)', 'HC_pvalue (null) (phi2)', 'HC (phi2)', 'HC_pvalue (phi2)', 'bonf (phi2)', 'F1 (phi2)', 'F1 (phi1)'])


    columns = ['title',  'edit_rate', 'length',
            f'HC (null) ({model1})', f'HC_pvalue (null) ({model1})', f'HC ({model1})', f'HC_pvalue ({model1})', f'bonf ({model1})',
            f'HC (null) ({model2})', f'HC_pvalue (null) ({model2})', f'HC ({model2})', f'HC_pvalue ({model2})', f'bonf ({model2})'
            ]

    df_disp = df_merged.filter(columns)

    def arrange_and_color(x: float, pvalue, color) -> str:
        # round x to 2 decilmal places and pvalue to 4 decimal places:
        if pvalue < 0.05:
            stat_str = f"{{\\color{{{color}}} {{{np.round(x,2)}}}}}"
            pvalue_str = f"\\textbf{{{np.round(pvalue, 4)}}}"
        else:
            stat_str = f"{np.round(x, 2)}"
            pvalue_str = str(np.round(pvalue, 4))
        return f"{stat_str} ({pvalue_str})"

    def arrange_and_color_red(x: float, pvalue) -> str:
        return arrange_and_color(x, pvalue, 'red')
    def arrange_and_color_blue(x: float, pvalue) -> str:
        return arrange_and_color(x, pvalue, 'blue')

    for model in [model1, model2]: # write Pvalues in the format stat (pvalue):
        df_disp[f'HC ({model})'] = df_disp.apply(lambda row: arrange_and_color_blue(row[f'HC ({model})'], row[f'HC_pvalue ({model})']), axis=1)
        df_disp[f'HC (null) ({model})'] = df_disp.apply(lambda row: arrange_and_color_red(row[f'HC (null) ({model})'], row[f'HC_pvalue (null) ({model})']), axis=1)

    # remove the pvalues from the dataframe:
    df_disp = df_disp.drop(columns=[f'HC_pvalue ({model1})', f'HC_pvalue ({model2})', f'HC_pvalue (null) ({model1})', f'HC_pvalue (null) ({model2})'])
    
    df_disp.columns = df_disp.columns.str.replace("_", "-")
    df_disp.loc[:, 'title'] = df_disp['title'].apply(lambda x: re.sub(r"([A-Z])([a-z]+) ([A-Z][a-z]+)", r"\1 \3", x))

    aa = df_disp.set_index('title')
    print(aa.to_latex(float_format=lambda x: '%.2f' % x))



if __name__ == "__main__":
    main()