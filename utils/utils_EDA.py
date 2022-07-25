import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

def plot_subplots(N, cols, figsize=(12,4)):
    """Useful function to plot multiple subplot given total number of plots and number of columns 
    args:
    N: Number of total plots
    cols: Number of columns in subplot"""
    rows = int(np.ceil(N/cols))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=figsize)
    return fig, gs

def plot_cat_violin(data, X_cat, y, n_cols, figsize=(12,4)):
    
    """Useful function to plot multiple categorical variables X_cat against target variable y 
    args:
    data: dataframe with all columns
    X_cat: list of categorical features to plot against
    y: name of target variable
    n_cols: Number of columns in subplot
    figsize: size of figure
    """

    fig, gs = plot_subplots(len(X_cat), n_cols, figsize=figsize)
    
    # Violin plots 
    for i, col in enumerate(X_cat):
        ax = fig.add_subplot(gs[i])
        sns.violinplot(x=col, y=y, data=data, ax=ax)
    return fig

def plot_num_violin(data, X_num, y, n_cols, figsize=(12,4)):
    
    """Useful function to plot multiple numerical variables X_num against target variable y 
    args:
    data: dataframe with all columns
    X_num: list of numerical features to plot against
    y: name of target variable
    n_cols: Number of columns in subplot
    figsize: size of figure
    """

    fig, gs = plot_subplots(len(X_num), n_cols, figsize=figsize)
    
    # Violin plots 
    for i, col in enumerate(X_num):
        ax = fig.add_subplot(gs[i])
        sns.violinplot(x=y, y=col, data=data, ax=ax)
    return fig

def plot_num_hist(data, X_num, y, n_cols, figsize=(12,4)):
    
    """Useful function to plot multiple numerical variables X_num against target variable y in a histogram
    args:
    data: dataframe with all columns
    X_num: list of numerical features to plot against
    y: name of target variable
    n_cols: Number of columns in subplot
    figsize: size of figure
    """

    fig, gs = plot_subplots(len(X_num), n_cols, figsize=figsize)
    
    # Violin plots 
    for i, col in enumerate(X_num):
        ax = fig.add_subplot(gs[i])
        for name, gr in data.groupby(y):
            sns.kdeplot(gr[col].dropna(), ax=ax)
    return fig
