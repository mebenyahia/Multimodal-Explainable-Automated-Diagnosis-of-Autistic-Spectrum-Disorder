import pandas as pd
import numpy as np
from itertools import chain
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

def load_data(file_path):
    """Load the dataset from the given file path."""
    data = pd.read_csv(file_path)
    return data

def find_duplicate_columns(df):
    """Find and return duplicate columns in the dataframe."""
    duplicate_columns = {}
    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if df[columns[i]].equals(df[columns[j]]):
                if columns[i] in duplicate_columns:
                    duplicate_columns[columns[i]].append(columns[j])
                else:
                    duplicate_columns[columns[i]] = [columns[j]]
    return duplicate_columns


def find_missing_value_indicators(df):
    indicators = {}
    flags = ['error', 'fail', '?']
    for column in df.columns:
        unique_values = df[column].unique()
        potential_indicators = []
        for value in unique_values:
            if (isinstance(value, str) and any(keyword in value.lower() for keyword in flags)):
                potential_indicators.append(value) 
        if potential_indicators:
            indicators[column] = potential_indicators
    return indicators

def remove_rows_with_elements_from_set(df, elements_set):
    indices_to_remove = set()
    for index, row in df.iterrows():
        for element in row:
            if element in elements_set:
                indices_to_remove.add(index)
                break  
    return df.drop(indices_to_remove)



def one_hot_encode(data):
    """Perform one-hot encoding on categorical variables."""
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cols = encoder.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    
    data.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    
    data = data.drop(columns=categorical_cols)
    data = pd.concat([data, encoded_df], axis=1)
    
    return data

def custom_one_hot_encode(data, column):
    data['HANDEDNESS_L'] = 0
    data['HANDEDNESS_R'] = 0

    data.loc[data[column] == 'R', 'HANDEDNESS_R'] = 1
    data.loc[data[column] == 'L', 'HANDEDNESS_L'] = 1
    data.loc[data[column].isin(['Ambi', 'Mixed']), ['HANDEDNESS_L', 'HANDEDNESS_R']] = 1
    
    return data

def preprocess_data(file_path):
    """Load the data and perform preprocessing steps."""

    data = load_data(file_path)
    print("Data loaded successfully.")
    
    #1: Setting all -9999 to nan
    data = data.replace([-9999.0, '-9999', 'no_filename'], np.NaN)
    data = data.dropna(subset='FILE_ID')


    #2: removing duplicate and extra identifier columns
    data.drop(columns=['subject', 'X', 'Unnamed: 0.1', 'Unnamed: 0', 'SITE_ID', 'FILE_ID'], inplace=True)


    #3: Remove erronous rows in QC columns
    qc_to_inspect = ['qc_rater_1', 'qc_notes_rater_1', 'qc_anat_rater_2', 'qc_anat_notes_rater_2', 'qc_func_rater_2', 'qc_func_notes_rater_2', 'qc_anat_rater_3', 'qc_anat_notes_rater_3', 'qc_func_rater_3', 'qc_func_notes_rater_3']
    potential_missing_value_indicators = find_missing_value_indicators(data)
    potential_missing_value_indicators = find_missing_value_indicators(data[qc_to_inspect])
    flags = set(chain(*potential_missing_value_indicators.values())) 
    data = remove_rows_with_elements_from_set(data, flags)

    #4: Remove QC + extra diagnosis column
    diag = ['DSM_IV_TR'] #this column gives the dx_group directly and would bias all models
    anatomical_qc = ['anat_cnr', 'anat_efc','anat_fber', 'anat_fwhm', 'anat_qi1', 'anat_snr']
    functional_qc = ['func_efc', 'func_fber','func_fwhm', 'func_dvars', 'func_outlier', 'func_quality', 'func_mean_fd', 'func_num_fd', 'func_perc_fd', 'func_gsr']
    raters_qc = ['qc_rater_1', 'qc_notes_rater_1', 'qc_anat_rater_2', 'qc_anat_notes_rater_2', 'qc_func_rater_2', 'qc_func_notes_rater_2', 'qc_anat_rater_3', 'qc_anat_notes_rater_3', 'qc_func_rater_3', 'qc_func_notes_rater_3', 'SUB_IN_SMP']
    qc = diag + anatomical_qc + functional_qc + raters_qc
    data = data.drop(columns=qc)

    #5: Optimization
    total_rows = len(data)
    def threshold(percentage, total_rows):
        return total_rows * percentage / 100

    def objective(percentage):
        percentage = percentage[0]  
        thresh = threshold(percentage, total_rows)
        num_cols = data.dropna(thresh=int(thresh), axis=1).shape[1]
        ratio_features_total = num_cols / total_rows
        ratio_features_non_missing = num_cols / thresh

        return - (ratio_features_total - ratio_features_non_missing)

    def constraint(percentage):
        percentage = percentage[0]  
        thresh = threshold(percentage, total_rows)
        num_cols = data.dropna(thresh=int(thresh), axis=1).shape[1]
        return total_rows - 2 * (num_cols-1)
    
    bounds = [(20, 80)]
    constraints = [{'type': 'ineq', 'fun': constraint}]
    initial_guess = [50]
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_percentage = result.x[0]
    optimal_thresh = threshold(optimal_percentage, total_rows)
    data = data.dropna(thresh=optimal_thresh, axis=1)

    #8: Dealing with handedness category
    #data.loc[:, 'HANDEDNESS_CATEGORY'] = data['HANDEDNESS_CATEGORY'].replace(['Ambi', 'Mixed'], np.NaN)
    #data = data.dropna(subset=['HANDEDNESS_CATEGORY'])
    
    data = custom_one_hot_encode(data, 'HANDEDNESS_CATEGORY')
    data = data.drop(columns=['HANDEDNESS_CATEGORY'])

    #6: Dealing with the other NaNs
    data.loc[:, 'CURRENT_MED_STATUS'] = data['CURRENT_MED_STATUS'].replace(['`'], np.NaN)
    data.drop(['FIQ_TEST_TYPE', 'PIQ_TEST_TYPE', 'VIQ_TEST_TYPE'], axis=1, inplace=True)
    data = data.dropna(subset=['FIQ', 'VIQ', 'PIQ'])

    data = one_hot_encode(data)
    
    data = data.set_index('SUB_ID')

    return data

if __name__ == "__main__":
    file_path = '/mnt/data/Phenotypic_V1_0b_preprocessed1.csv'
    processed_data = preprocess_data(file_path)
    processed_data.to_csv('/mnt/data/processed_data.csv', index=True)
    print("Data preprocessing complete. Processed data saved to '/mnt/data/processed_data.csv'.")

