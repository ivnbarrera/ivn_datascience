from time import sleep
import datetime as dt
import pandas as pd
import numpy as np
import math
import itertools
import re
from itertools import compress
from nltk.corpus import wordnet
import nltk
import matplotlib.pyplot as plt
import os
import dedupe
import csv


def remove_duplicates(df, sort_columns, column_id_name):
    """
    Function that removes duplicate based on sort columns
    Parameters are pieces of information to update
    """
    df_original = df.copy()
    if np.all([column in df_original.columns for column in sort_columns]):
        
        # drop address columns
        adresses_columns = ['address','city','state','zip_code']
        has_address_col = [True if col in str(sort_columns) else False for col in adresses_columns]
        if np.all(has_address_col):
                        
            address_col = [col for col in sort_columns if 'address' in col][0]
            city_col  = [col for col in sort_columns if 'city' in col][0]
            state_col  = [col for col in sort_columns if 'state' in col][0]
            zip_code_col  = [col for col in sort_columns if 'zip_code' in col][0]
            
            adresses_columns = [address_col,city_col,state_col,zip_code_col]
            sort_columns = [col for col in sort_columns if col not in adresses_columns]
            
            df_original['hasinfo_address'] = df_original[address_col].notnull() & df_original[city_col].notnull() \
                    & df_original[state_col].notnull() & df_original[zip_code_col].notnull()
        else:
            print('Full address not contemplated. To update complete address include: {}'.format(
                ",".join(list(compress(adresses_columns, np.logical_not(in_append))))
            ))
        name_columns = ['name','first_name','last_name']
        has_name_col = [True if col in str(sort_columns) else False for col in name_columns]
        if np.all(has_name_col):
            
            name = [col for col in sort_columns if 'name' == col][0]
            fn_col  = [col for col in sort_columns if 'first_name' in col][0]
            ln_col  = [col for col in sort_columns if 'last_name' in col][0]
            
            name_columns = [name,fn_col,ln_col]
            sort_columns = [col for col in sort_columns if col not in name_columns]
            
            df_original['hasinfo_name'] = df_original[fn_col] + ' ' + df_original[ln_col]  == df_original[name]
        email_columns = ['email','merlin_email']
        has_email_col = [True if col in str(sort_columns) else False for col in email_columns]
        if np.all(has_email_col):
            
            merlin_email  = [col for col in sort_columns if 'merlin_email' in col][0]
            email  = [col for col in sort_columns if 'email' in col][0]
            
            email_columns = [email,merlin_email]
            sort_columns = [col for col in sort_columns if col not in email_columns]
            
            df_original['hasinfo_email'] = df_original[email] == df_original[merlin_email]
        
        has_name_col = [True if col in str(sort_columns) else False for col in name_columns]
        for column in sort_columns:
            df_original['hasinfo_'+column] = df_original[column].notnull()
        
        del_cols = [col for col in df_original if "hasinfo_" in col]
        df_original = df_original.sort_values(by=del_cols,
                                              ascending=[False for col in del_cols]
                                             )
        
        df_original.drop(columns=del_cols, inplace=True)
        df_original = df_original.drop_duplicates(subset=[column_id_name])
        return df_original
        print('success')
    else: print('columns not found in DF: \n original: {}'.format(
        ",".join([col for col in sort_columns if col not in df_original.columns])
    )) 
    return df_original

def append_info(df_original, df_append, ID_column, append_columns, mode='left'):
    """
    Function that Enrich contact information if a duplicate is found
    If the contact already had information, keep it, if not use new information from duplicate
    Parameters are pieces of information to update
    """
    join_mode = 'outer' if mode=='full' else 'left' if mode=='left' else 'left'
    if np.all([column in df_original.columns for column in append_columns]) & np.all([column in df_append.columns for column in append_columns]):
        
        df_original = pd.merge(
            df_original,
            df_append,
            how=join_mode,
            on=ID_column,
            suffixes=["","_append"]
        )
        
        # drop address columns
        adresses_columns = ['address','city','state','zip_code']
        in_append = [True if col in str(append_columns) else False for col in adresses_columns]
        if np.all(in_append):
            
            address_col = [col for col in append_columns if 'address' in col][0]
            city_col  = [col for col in append_columns if 'city' in col][0]
            state_col  = [col for col in append_columns if 'state' in col][0]
            zip_code_col  = [col for col in append_columns if 'zip_code' in col][0]
            
            adresses_columns = [address_col,city_col,state_col,zip_code_col]
            append_columns = [col for col in append_columns if col not in adresses_columns]
            
            complete_address = pd.notnull(df_original[adresses_columns[0]]) & pd.notnull(df_original[adresses_columns[1]])\
                                & pd.notnull(df_original[adresses_columns[2]]) & pd.notnull(df_original[adresses_columns[3]])
            new_complete_address = pd.notnull(df_append[adresses_columns[0]]) & pd.notnull(df_append[adresses_columns[1]])\
                                & pd.notnull(df_append[adresses_columns[2]]) & pd.notnull(df_append[adresses_columns[3]])
            #return((complete_address,new_complete_address))
            for col in adresses_columns:
                df_original.loc[~complete_address & new_complete_address, col] = df_original[col+'_append']
        else:
            print('Full address not contemplated. To update complete address include: {}'.format(
                ",".join(list(compress(adresses_columns, np.logical_not(in_append))))
            ))
        for column in append_columns:
            df_original[column] = df_original[column].fillna(df_original[column+'_append'])
        df_original.drop(columns=[col for col in df_original if "_append" in col], inplace=True)
        print('success')
    else: print('columns not found in DF: \n original: {}, \n append: {}'.format(
        ",".join([col for col in append_columns if col not in df_original.columns]),
        ",".join([col for col in append_columns if col not in df_append.columns]),
    )) 
    return df_original

def deduping_function(fields, training_file, settings_file,
                      input_dict, threshold=0.5,
                      active_labeling=True, pairs=None, save_settings=True
                      ):
    """
    Function to dedupe rows given a
    fields: list of dictionaries of fields relevant for the training
    training_file: json with training set if the function was already run
    settings_file: file with settings if the function was already run
    input_file: CSV with values to dedup
    output_file: CSV where results are going to be written
    input_dict: dictionary with keys as id and values as dictionary of features for each row
    
    Result is to write the output_file
    """
    if os.path.exists(settings_file):
        print('reading from {}'.format(settings_file))
        with open(settings_file, 'rb') as f:
            deduper = dedupe.StaticDedupe(f)
    else:
        deduper = dedupe.Dedupe(fields)
        deduper.prepare_training(data)
        if os.path.exists(training_file):
            print ('reading labeled examples from ',format(training_file))
            with open(training_file) as tf:
                deduper.readTraining(tf)

        if active_labeling:
            print('starting active labeling...')
            dedupe.console_label(deduper)
        else:
            if not pairs:
                raise AssertionError("If not active_labeling you must provide pairs")

        deduper.train()

        if save_settings:
            print('writing training file')
            with open(training_file, 'w') as tf:
                deduper.writeTraining(tf)
            print('writing settings file')
            with open(settings_file, 'wb') as sf:
                deduper.writeSettings(sf)

    # print('blocking...')
    # threshold = deduper.threshold(input_dict, recall_weight=2)
    print ('clustering...')
    clustered_dupes = deduper.partition(input_dict, threshold)
    print ('# duplicate sets {}'.format(len(clustered_dupes)))
    
    cluster_membership = {}
    for cluster_id, (records, scores) in enumerate(clustered_dupes):
        rcluster = [data[c] for c in records]
        canonical_rep = dedupe.canonicalize(rcluster)
        for record_id, score in zip(records, scores):
            cluster_membership[record_id] = {
                "Cluster ID": cluster_id,
                "confidence_score": score,
                "canonical_rep": canonical_rep,
            }

    print('deduping sucessfully finished')
    return cluster_membership