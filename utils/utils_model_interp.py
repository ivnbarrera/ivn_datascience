import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def svc_boundaries_binary(model, X, y, ax):
    """
    Function to plot decision boundaries for Support Vector Classifier on binary classification
    :params
        model - trained SVC model
        X - features DF
        y - list of labeled examples with matching index to X
        ax - axis to plot the figure
    """
    distances = pd.DataFrame(model.decision_function(X), index=X.index)
    colors = ['b','r']
    # For each label plot the distribution in same graphic
    for l, c in zip(np.unique(y), colors):
        sns.distplot(distances[y == l], color=c, kde=False, ax = ax, norm_hist=True)

    ax.legend(np.unique(y))
    # limit to only relevant examples
    ax.set_xlim((-2,2))
    # Set reference lines
    ax.axvline(x=1, linestyle='dashed', linewidth=0.8, c='black')
    ax.axvline(x=-1, linestyle='dashed', linewidth=0.8, c='black')
    ax.axvline(x=0, linestyle='dashed', linewidth=1, c='black')
    ax.set_title('Histogram of Projections \n Separation From Decision Boundary')
    
def synt_probabilities(X, model, x_min, x_max, x_mean, cols, cols_non_cat, resolution=100):
    """
    Function that return sensitivity probabilities given some columns to evaluate
    :params
        X - original features data to evaluate
        model - trained model with predict_proba available
        x_min - Dataframe with minimum values for columns
        x_max - Dataframe with maximum values for columns
        x_mean - Dataframe with mean values for columns
        cols - columns to evaluate
        cols_non_cat - list of columns which are not categorical
    :returns
        synthetic_probabilities - dictionary of synthetic probabilities, each key is a column
    """
    # Initialize return dictionary
    synthetic_probabilities = {}
    
    for col in cols:
        # If the column is not categorical use min and maximum to create range
        if col in cols_non_cat:
            # Get minimum, maximum and range for specific column
            # [0] is used to get the actual value not a Series with the value
            x_min_col = x_min[col][0]
            x_max_col = x_max[col][0]
            x_range_col = x_max_col - x_min_col
            # Create Range
            synthetic_range = np.arange(start=x_min_col,stop=x_max_col,step=x_range_col/resolution)
            # Duplicate x_mean DF by resolution times
            X_synthetic = pd.concat([x_mean]*resolution, ignore_index=True)
        # If column is categorical use categories to create range
        else:
            # Get unique labels in category
            synthetic_range = X[col].unique()
            len_col = len(synthetic_range)
            # Duplicate x_mean DF by number of categories times
            X_synthetic = pd.concat([x_mean]*len_col, ignore_index=True)
        # Replace column with synthetic range
        # This way all other columns are held constant except the column being evaluated
        X_synthetic[col] = synthetic_range
        # Predict on new syntethic samples
        try:
            class_probabilities = model.predict_proba(X_synthetic)
            # Save results in dict
            synthetic_probabilities[col] = pd.DataFrame(index=synthetic_range,data=class_probabilities)
        except:
            raise Exception('Model does not have predict_proba method') 
    return synthetic_probabilities

def avg_synt_proba(X, y, model, cols, cols_non_cat, positive_only=False):
    """
    Function that returns average sensitivity for selected columns given a model and samples
    :params
        X - original features data to evaluate
        model - trained model with predict_proba available
        cols - columns to evaluate
        cols_non_cat - list of columns which are not categorical
        run_type - if only positives samples should be used in the sensitivity analysis
    :returns
        avg_synthetic_probabilities - dictionary of average synthetic probabilities, each key is a column
    """
    X_stats = X.dropna().describe()
    # If only positive examples get only labeled examples = 1
    if positive_only: X_eval = X[y==1].dropna()
    else: X_eval = X.dropna()
    # Initialize synthetic probabilities   
    synth_probs = []
    # Get synthetic probabilities for each sample to get a better understanding
    for i in range(X.shape[0]):
        x_mean = pd.DataFrame(X.iloc[i,:]).transpose()
        x_max = pd.DataFrame(X_stats.loc['max',:]).transpose()
        x_min = pd.DataFrame(X_stats.loc['min',:]).transpose()
        # Get syntetic probabilities
        synth_probs_i = synt_probabilities(X,model,x_min, x_max, x_mean, cols,['Age','Parch'])
        # Save results to list
        synth_probs.append(synth_probs_i)
    # Sum over all sensitivity distributions
    ## Initialize sum of probabilities
    summed_synthetic_probabilities = synth_probs[0] 
    for i in range(1,len(synth_probs)):
        for key in summed_synthetic_probabilities:
            # Accumulate probability 
            summed_synthetic_probabilities[key] += synth_probs[i][key]
    # Calculate an average agreement of feature sensitivity
    ## Initialize avg of probabilities
    avg_synthetic_probabilities = {}
    for key in summed_synthetic_probabilities:     
        avg_synthetic_probabilities[key] = summed_synthetic_probabilities[key] / len(synth_probs)
    return avg_synthetic_probabilities