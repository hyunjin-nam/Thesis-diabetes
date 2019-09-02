
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd

def devide_data(df):
    features = list(df.columns.drop('y'))
    y = df['y']
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Divide by class
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    return X_train, X_test, y_train, y_test, df_train, df_test


######################
###### Resample ######
######################

def imbalance_oversample(df_train):
### Oversample
# Class count
    count_class_0, count_class_1 = df_train.y.value_counts()
    df_class_0 = df_train[df_train['y'] == 0]
    df_class_1 = df_train[df_train['y'] == 1]
    df_class_1_sample = resample(df_class_1,  replace=True, n_samples=len(df_class_0), random_state=27) # reproducible results


    sample_train = pd.concat([df_class_0, df_class_1_sample])
    y_train = sample_train.y
    X_train = sample_train.drop('y', axis=1)
    print('Data is re-sampled with using Over sampling')
    return X_train, y_train;


def imbalance_undersample(df_train):
    ### Undersample
    count_class_0, count_class_1 = df_train.y.value_counts()
    df_class_0 = df_train[df_train['y'] == 0]
    df_class_1 = df_train[df_train['y'] == 1]
    df_class_0_sample = resample(df_class_0,  replace=False, n_samples=len(df_class_1), random_state=27) # reproducible results


    sample_train = pd.concat([df_class_0_sample, df_class_1])
    y_train = sample_train.y
    X_train = sample_train.drop('y', axis=1)
    print('Data is re-sampled with using Under sampling')
    return X_train, y_train;


