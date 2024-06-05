import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.stats import norm
mpl.style.use('ggplot')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score

#folder = "results"
#folder = "/Users/kipnisal/Google Drive/results"
FOLDER = "results"

datasets = ["abstracts", "news", "wiki-long"]

#model = "gpt2"
#model = "gpt-neo-1.3B"
model = "gpt2-xl"
#model = "falcon-7b"
#model = "llama-13b"

BITS = True
MAX_TOKENS_PER_SENTENCE = 50
MIN_TOKENS_PER_SENTENCE = 10
IGNORE_FIRST_SENTENCE = True
context = "no_context"


def get_datar(dataset, model, context):
    name = f"{model}_{context}_{dataset}"
    data_machine = pd.read_csv(f"{FOLDER}/{name}_machine.csv")
    data_human = pd.read_csv(f"{FOLDER}/{name}_human.csv")

    if dataset == "news" and model == "gpt-neo-1.3B":
        data_human['length'] /= 5
        print("Adjusting char to tokens")

    if BITS == True:
        data_human['response'] /= np.log(2)
        data_machine['response'] /= np.log(2)


    datar = pd.concat([data_machine, data_human])
    if dataset == 'wiki' or dataset == 'wiki-long':
        human_author =  'Wikipedia'
        machine_author = 'GPT3'
        data_machine['author'] = 'GPT3'
    if dataset == 'news':
        human_author =  'Human'
        machine_author = 'ChatGPT'
    if dataset == 'abstracts':
        human_author =  'Sc. Abstracts'
        machine_author = 'ChatGPT'

    data_machine['author'] = machine_author
    data_human['author'] = human_author

    datar = pd.concat([data_machine, data_human])

    datar = datar[(MIN_TOKENS_PER_SENTENCE <= datar.length) & (datar.length <= MAX_TOKENS_PER_SENTENCE)]
    datar = datar.groupby('response').head(1) # remove repeated entries
    datar = datar.rename(columns={'sent_num': 'num'})


    if context == 'previous_sentence' or IGNORE_FIRST_SENTENCE:
        datar = datar[datar.num>1]

    return datar, {'human_author': human_author, 'machine_author': machine_author}

def plot_histogram(dataset, model, context):
    
    datar, authors = get_datar(dataset, model, context)

    tt = np.linspace(0,10,157)
    datar.groupby('author').response.plot.hist(bins = tt, density=True, alpha=.7, legend=True, 
                                                color={authors['machine_author'] : 'tab:red',
                                                       authors['human_author'] :'tab:blue'})
    plt.legend(fontsize=16)
    plt.ylabel('')
    #plt.xlabel('log-ppt [bits/token]', size=20)

    plt.xlim((0,10))
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    
    

def plot_ROC(dataset, model, context):

    datar, authors = get_datar(dataset, model, context)

    fpr, tpr, _ = roc_curve(y_true=datar.author, y_score=datar.response, pos_label=authors['human_author'])
    mdr = 1 - tpr
    istar = np.argmin(fpr + mdr)
    Rstar = fpr[istar] + mdr[istar]
    
    alpha = 0.05
    
    beta = 1 - tpr[fpr <= alpha].max()
    Del = norm.isf(alpha) - norm.ppf(beta)  # effect size on the Z-scale
    y_true = (datar.author == authors['human_author'])
    auc = roc_auc_score(y_true=y_true, y_score=datar.response)

    res = dict(name=name, alpha=alpha, beta=beta, delta=Del,
            Rstar = Rstar,
            AUC=auc)


    display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    display.plot()
    plt.plot([0, 1], [0, 1], "k", alpha=.25)

    #plt.scatter(Rsxy[0], Rsxy[1], c='r')
    #plt.scatter(A005xy[0], A005xy[1], c='r')

    plt.axis("square")
    plt.xlabel("", fontsize=16)
    plt.ylabel("", fontsize=16)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    
    
    #plt.legend([],fontsize=20)
    plt.text(0,0.90,s=f"AUC = {np.round(res['AUC'],3)}", size=16, backgroundcolor='w')
    #plt.text(0,0.80,s=f"$\Delta_z({alpha}) = {np.round(res['delta'],3)}$", backgroundcolor='w', size=20)
    #plt.title("sentence detector")
    return plt.gcf()
    
    
for dataset in datasets:
    name = f"{model}_{context}_{dataset}"
    # fig = plt.figure()
    # fig.set_size_inches(8, 6, forward=True)
    # plot_histogram(dataset, model, context)
    # fig.savefig(f"Figs/histogram_{name}.png")

    g = plot_ROC(dataset, model, context)
    g.savefig(f"Figs/ROC_{name}.png")