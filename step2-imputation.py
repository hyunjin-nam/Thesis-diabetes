from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
import pandas as pd
import random
import numpy as np


# Impute with 0
def impute_zero(df):
    df = df.fillna(0)
    print('Data is imputed with using 0 Imputation')
    return(df);

# Knn
def impute_knn(df):
    y = df.y
    df_target = df.drop('y', axis=1)
    df_filled_knn = KNN(k=3, verbose=False).fit_transform(df_target);
    df_filled_knn = pd.DataFrame(df_filled_knn)
    df_filled_knn.columns = df_target.columns
    df_filled_knn['y'] = y
    df = df_filled_knn
    print('Data is imputed with using Knn Imputation')
    return(df);



# Impute with median
def impute_median(df):
    df = df.fillna(df.median())
    print('Data is imputed with using Median Imputation')
    return(df);

# Impute with mean
def impute_mean(df):
    df = df.fillna(df.mean())
    print('Data is imputed with using Mean Imputation')
    return(df);

# Impute with random
def impute_random(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.random.choice(df[col].dropna().values) if np.isnan(x) else x)
    print('Data is imputed with using Random Imputation')
    return(df);


