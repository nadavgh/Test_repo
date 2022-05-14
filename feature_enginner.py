import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
import os
from data_analysis import Nans_count #, data_collector
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# We will use this mean to replace missing ualues in the training and test set.
def mean_imputation(patient: pd.DataFrame, missing: str):
    mean_non_missing = patient[missing].mean()
    missing_col = patient[missing]
    missing_col.fillna(mean_non_missing, inplace=True)


# We will use this median to replace missing ualues in the training and test set.
def median_imputation(patient: pd.DataFrame, missing: str):
    mean_non_missing = patient[missing].median()
    missing_col = patient[missing]
    missing_col.fillna(mean_non_missing, inplace=True)


# Linear interpolation for missing data series
def backward_fill_imputation(patient: pd.DataFrame, missing: str):
    # missing_col = patient[missing]
    # first_value = list(missing_col.isna()).index(True)
    # if first_value > 0:
    #     missing_col[missing_col.index < first_value] = first_value
    # for
    patient[missing].interpolate(limit_direction="both", inplace=True)
    return patient


def delete_cols(df: pd.DataFrame, tresh=95):
    remove = []
    nans = Nans_count(df)
    for feature in nans.keys():
        if nans[feature] > tresh:
            remove.append(feature)  # TODO there is a better way to do it since nans is ordered
    print('The sum of the rows we removing from the data: ', df[[c for c in df.columns if c in remove]].notna().sum())
    print('Number of features removed: ', len(remove))

    return df[[c for c in df.columns if c not in remove]]


def input_from_file(path, model):
    df = pd.read_csv(path, sep='|')
    id = str(path).split("_")[1].split(".")[0]
    df['id'] = len(df) * [int(id)]
    if 1 in list(df['SepsisLabel']):
        idx = list(df['SepsisLabel']).index(1)
        if model == 'baseline':
            df = df.iloc[idx]
        else:
            df = df[1:idx + 1]  # TODO Maybe make all labels 1
    elif model == 'baseline':
        df = df.iloc[-1]
    return df[1:]


def get_input(directory: str, model='Advanced'):
    if directory not in ['train', 'test']:
        raise ValueError('Directory must be `train` or `test`.')
    input_df = pd.DataFrame()
    for file in tqdm(os.scandir('data/' + directory)):
        file_input = input_from_file(file, model=model)
        input_df = input_df.append(file_input)
        if len(input_df) > 500:
            break
    return input_df


def clean_data(df: pd.DataFrame):
    # df = data_collector(phase='train')
    nulls = Nans_count(df)
    df = delete_cols(df, tresh=98)
    descb = pd.read_csv('Statistical_stuff.csv', index_col='Unnamed: 0')
    tresh_null, tresh_std = 70, 7.5
    for col in tqdm(df.columns):
        for idx in df['id']:
            patient = df[df['id'] == idx]
            if 0 < nulls[col] < tresh_null:
                filled_patient = backward_fill_imputation(patient, col)
            else:
                if descb[col]['std'] < tresh_std:
                    filled_patient = mean_imputation(patient, col)
                else:
                    filled_patient = median_imputation(patient, col)
            df[df['id'] == idx] = filled_patient
    print()


def main():
    df = get_input('train')
    clean_df = clean_data(df)
    print()


if __name__ == '__main__':
    main()

# TODO decide which rows to fill with which method
# TODO first row all nulls what we do with it