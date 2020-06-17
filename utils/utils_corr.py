import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from scipy.stats.stats import pearsonr, ttest_ind_from_stats, ttest_ind
from dython.nominal import theils_u
import itertools

def cramers_v(x, y):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def theils_u(x, y):
    return theils_u(x, y)

def fix_snsheatmap_bug():
    """
    This function just fix a temporary bug on sns where top and bottom are cut in half
    
    """
    ## --- temporary fix bug where top and bottom rows are cut in half
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t)
    ## --- temporary fix bug where top and bottom rows are cut in half

def get_numcorr_heatmap(data, cols=None, figsize=(10,10), method='pearson', cmap="RdYlGn"):
    """
    Plot a heatmap or correlation for numerical variables
    :param 
        data - DF with relevant columns
        cols - list of numerical columns to evaluate
        figsize - size of the pyplot figure
        method - correlation method to evaluate cols
        cmap - color of heatmap
    """
    if cols is not None:
        corr_df = data[cols]
    else:
        corr_df = data
    # Get correlation coefficients for all columns
    corrmat = corr_df.corr(method=method)
    top_corr_features = corrmat.index
    fig = plt.figure(figsize=figsize)
    #Plot heat map using anotations
    ax = sns.heatmap(corr_df[top_corr_features].corr(method=method),annot=True,cmap=cmap)
    fix_snsheatmap_bug()
    ax.set_title("{} Correlation between Variables".format(method))
    plt.show()

def get_catcorr_heatmap(data, cols=None, figsize=(10,10), method='cramersV', cmap="RdYlGn"):
    """
    Plot a heatmap or correlation for numerical variables
    :param 
        data - DF with relevant columns
        cols - list of categorical columns to evaluate
        figsize - size of the pyplot figure
        method - correlation method to evaluate cols
        cmap - color of heatmap
    """
    if method == 'cramersV': func = cramers_v
    elif method == 'theilsU': func = theils_u

    if cols is not None: n = m = len(cols)
    else: n = m = len(data.columns)
        
    corrM = np.zeros((n,m))
    
    for col1, col2 in itertools.combinations(cols, 2):
        idx1, idx2 = cols.index(col1), cols.index(col2)
        corrM[idx1, idx2] = func(data[col1], data[col2])
        corrM[idx2, idx1] = corrM[idx1, idx2]

    corr = pd.DataFrame(corrM, index=cols, columns=cols)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(corr, annot=True, ax=ax, cmap=cmap)
    fix_snsheatmap_bug()
    ax.set_title("{} Correlation between Variables".format(method))
    plt.show()

def supected_num_pairs(data,pairs):
    """
    Plot a heatmap or correlation for numerical variables
    :param 
        data - DF with relevant columns
        pairs - list of tuples with suspected columns pairs
    :returns
        result: DataFrame with correlation coefficient and p-value
    """
    result = pd.DataFrame(columns=['col1','col2','pearsonr','pvalue'])
    for col1, col2 in pairs:
        data_pair = data.dropna(subset=[col1, col2])
        coeff, p = pearsonr(data_pair[col1], data_pair[col2])
        to_append = pd.DataFrame([{'col1':col1,'col2':col2,'pearsonr':coeff,'pvalue':p}])
        result = result.append(to_append, sort=False, ignore_index=True)
    return result

def ttest_variables_targetBinary(data, X, y=None, method='data'):
    """
    Calculates the effect of feature variables on target variable
    using ttest and save it on a DataFrame
    :param
        data - DataFrame with relevant columns or dictionary with statistics (mean, std)
        X - feature columns
        y - target variable name,  only binary y is accepted for now
        method - ttest to be used, 'either' data or 'statistics' (when only mean and std available)
    :return
        result - DataFrame with column name, statistic value and p-value
    """
    result = pd.DataFrame(columns=['col','statistic_value','pvalue'])
    if method == 'data':
        assert isinstance(data, pd.DataFrame), "Data is not a DataFrame"
        assert y is not None, "Target variable name not provided"
        y_levels = data[y].unique()
        assert len(y_levels) == 2, "More that two levels in target variable"
        for col in X:
            data_col = data.dropna(subset=[col])
            level_0 = data_col.loc[data_col[y] == y_levels[0], col]
            level_1 = data_col.loc[data_col[y] == y_levels[1], col]
            stat, p = ttest_ind(level_0, level_1)
            to_append = pd.DataFrame([{'col':col,'statistic_value':stat,'pvalue':p}])
            result = result.append(to_append, sort=False, ignore_index=True)
    elif method == 'statistics':
        assert isinstance(data, dict), "Data is not a Dictionary"
        for col in X:
            assert col in data, "{} not in data".format(col)
            data_col = data[col]
            stats_ = ['mean1', 'mean2','std1','std2','nob1','nob2']
            for stat in stats_:
                assert stat in data_col, "{} not in {} data".format(stat, col)
            mean1, std1, nob1 = data_col['mean1'], data_col['std1'], data_col['nob1']
            mean2, std2, nob2 = data_col['mean1'], data_col['std2'], data_col['nob2']
            stat, p = ttest_ind_from_stats(mean1, std1, nob1, mean2, std2, nob2)
            to_append = pd.DataFrame([{'col':col,'statistic_value':stat,'pvalue':p}])
            result = result.append(to_append, sort=False, ignore_index=True)
    return result