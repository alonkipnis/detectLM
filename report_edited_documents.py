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
    # two models
    parser.add_argument('-edited-results-file1', type=str, help='file with results of edited documents model 1', default="")
    parser.add_argument('-edited-results-file2', type=str, help='file with results of edited documents model 2', default="")

    parser.add_argument('-null-results-file1', type=str, help='file with results of non-edited documents 1', default="") 
    parser.add_argument('-null-results-file2', type=str, help='file with results of non-edited documents 2', default="") 
    

    args = parser.parse_args()

    if args.edited_results_file1 != "" and args.edited_results_file2 != "":
        df_disp1 = arrange_results(args.edited_results_file1, args.null_results_file1)
        df_disp2 = arrange_results(args.edited_results_file2, args.null_results_file2)
        merge_results_of_two_models(df_disp1, df_disp2, model1="gpt2", model2="phi2")
        return
    else:
        results_edited_filename = args.edited_results_file
        results_null_filename = args.null_results_file
        df_disp = arrange_results(results_edited_filename, results_null_filename)
        arrange_and_print_one_model(df_disp)
        

def arrange_results(results_edited_filename, results_null_filename):
    df = pd.read_csv(results_edited_filename)
    # get basename of the file, and extract the title of the document.
    basenames = df.iloc[:,0].apply(os.path.basename)
    # now extract the name of the document
    df.loc[:, 'title'] = [re.findall(r"([A-Za-z \(\)]+)_edited[1-10]?\.*", x)[0] for x in basenames]
    logging.info(f"Loaded {len(df)} rows from {results_edited_filename}")

    df_null = pd.read_csv(results_null_filename)
    # first extract the basename of the file:
    basenames = df_null.iloc[:,0].apply(os.path.basename)
    # now extract the name of the document
    df_null.loc[:, 'title'] = [re.findall(r"([A-Za-z \(\)]+)\.*", x)[0] for x in basenames]
    logging.info(f"Loaded {len(df_null)} rows from {results_null_filename}")

    df_disp = df.merge(df_null, on = 'title', suffixes=["", " (null)"], how='inner')
        
    print("Could not find a match to the following titles:")
    #print(df_disp[df_disp.filter(['title', 'length']).isna().any(axis=1)]['title'])

    df['F1'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])
    df['TPR'] = df['recall']

    columns = ['title',  'edit_rate', 'length', 'HC', 'HC_pvalue', 'HC (null)',  'HC_pvalue (null)',  'bonf', 'bonf (null)', 'F1']
    
    return df_disp.filter(columns)


def merge_results_of_two_models(df_disp1, df_disp2, model1, model2):
    """
    Merge the results of two models into a single dataframe, and print the results as a latex table
    """
    
    df_merged = df_disp1.merge(df_disp2, on = ['title', 'edit_rate', 'length'], suffixes=[" (gpt2)", " (phi2)"], how='outer')

    df_merged.to_csv("publich_bio_merged.csv")

    columns = ['title',  'length', 'edit_rate', 
            f'HC (null) ({model1})', f'HC_pvalue (null) ({model1})', f'HC ({model1})', f'HC_pvalue ({model1})', f'bonf ({model1})', f'bonf (null) ({model1})',
            f'HC (null) ({model2})', f'HC_pvalue (null) ({model2})', f'HC ({model2})', f'HC_pvalue ({model2})', f'bonf ({model2})', f'bonf (null) ({model2})',
    ]
    df_disp = df_merged.filter(columns)

    # def round_and_bold_blue(x):
    #     return f"\\color{{blue}} \\textbf{{{np.round(x,3)}}}"
        
    # def round_and_bold_red(x):
    #     return f"\\color{{red}} \\textbf{{{np.round(x,3)}}}"

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
    def add_color_scaler(x: float, color='blue') -> str:
        if x < 0.05:
            return f"{{\\color{{{color}}} {{{np.round(x,2)}}}}}"
        else:
            return f"{np.round(x, 2)}"
        

    for model in [model1, model2]: # write Pvalues in the format stat (pvalue):
        df_disp[f'HC ({model})'] = df_disp.apply(lambda row: arrange_and_color_blue(row[f'HC ({model})'], row[f'HC_pvalue ({model})']), axis=1)
        df_disp[f'HC (null) ({model})'] = df_disp.apply(lambda row: arrange_and_color_red(row[f'HC (null) ({model})'], row[f'HC_pvalue (null) ({model})']), axis=1)

        df_disp[f'bonf ({model})'] = df_disp[f'bonf ({model})'].apply(add_color_scaler, color='blue')
        df_disp[f'bonf (null) ({model})'] = df_disp[f'bonf (null) ({model})'].apply(add_color_scaler, color='red')

    # remove the pvalues from the dataframe:
    df_disp = df_disp.drop(columns=[f'HC_pvalue ({model1})', f'HC_pvalue ({model2})', f'HC_pvalue (null) ({model1})', f'HC_pvalue (null) ({model2})'])

    df_disp.columns = df_disp.columns.str.replace("_", "-")

    # only keep the last name:
    df_disp.loc[:, 'title'] = df_disp['title'].apply(lambda x: x.split()[-1])

    aa = df_disp.set_index('title')
    print(aa.to_latex(
        float_format=lambda x: '%.2f' % x
        ))

def arrange_and_print_one_model(df_disp):

        def round_and_bold_blue(x):
            return f"\\color{{blue}} \\textbf{{{np.round(x,3)}}}"
        def round_and_bold_red(x):
            return f"\\color{{red}} \\textbf{{{np.round(x,3)}}}"

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




if __name__ == "__main__":
    main()