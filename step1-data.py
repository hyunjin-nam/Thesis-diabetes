import pandas as pd
import re

input_file = "/Users/namhyunjin/PycharmProjects/untitled2/synthea_validate.csv"

df = pd.read_csv(input_file, header=0)
df_counts = df.Condition.value_counts()

def data_y(df, disease):

    df['y'] = df.Condition.isin(disease)

    d_tf = {True:1, False:0}
    df['y'] = df['y'].map(d_tf)

    dff = df[df.y == True]
    nff = df[df.y == False]

    mdff = pd.isna(dff)
    mnff = pd.isna(nff)

    mdff.mean(axis = 0, skipna = True)
    mnff.mean(axis = 0, skipna = True)

    mdiff = {"y" : mdff.mean(axis = 0, skipna = True), "not-y" : mnff.mean(axis = 0, skipna = True) }
    mdiff = pd.DataFrame(mdiff)

    # Exclude variables that all of the values are missing among y patients
    ex1 = mdiff[mdiff.y == 1].axes
    mdiff = mdiff.drop(ex1[0])
    df = df.drop(ex1[0],axis=1)


    # Blood_Pressure (Slit it into two variables which are high blood pressure and low blood pressure)
    low_blood = []
    high_blood = []
    for row in df['Blood_Pressure']:
        if pd.isna(row) == False:
            high_blood.append(float(re.findall('\d+', row)[1]))
            low_blood.append(float(re.findall('\d+', row)[0]))
        else:
            high_blood.append(float('Nan'))
            low_blood.append(float('Nan'))
    df['high_blood'] = low_blood
    df['low_blood'] = high_blood

    df = df.drop(df.dtypes[df.dtypes == "object"].axes[0], axis =1)
    df = df.drop('Unnamed: 0', axis =1)
    df = df.select_dtypes(exclude=['object'])


    # rename variables
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    a = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]
    df.columns = a

    df['y'].value_counts() #13410 522 #13003 929
    return(df);



df = data_y(df, ['Diabetes', 'Prediabetes'])
