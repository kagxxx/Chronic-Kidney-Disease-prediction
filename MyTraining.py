import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import random

df = pd.read_csv('kidney_disease.csv')

columns = pd.read_csv(r"data_description.txt", sep='-')
columns = columns.reset_index()

columns.columns=['cols','abb_col_names']

df.columns=columns['abb_col_names'].values

def convert_dtype(df, feature):
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

features = ['packed cell volume', 'white blood cell count', 'red blood cell count']
for feature in features:
    convert_dtype(df, feature)

df.drop('id', axis=1, inplace=True)

def extract_cat_num(df):
    cat_col = [col for col in df.columns if df[col].dtype == 'object']
    num_col = [col for col in df.columns if df[col].dtype != 'object']
    return cat_col, num_col


cat_col, num_col = extract_cat_num(df)


df['diabetes mellitus'].replace(to_replace={'\tno':'no', '\tyes':'yes', ' yes':'yes'}, inplace=True)
df['coronary artery disease'].replace(to_replace={'\tno':'no'}, inplace=True)
df['class'] = df['class'].replace(to_replace='ckd\t', value='ckd')


data = df.copy()


def assigning_missing_values(feature):
    random_sample=data[feature].dropna().sample(data[feature].isnull().sum())
    random_sample.index = data[data[feature].isnull()].index
    data.loc[data[feature].isnull(), feature] = random_sample


for col in num_col:
    assigning_missing_values(col)

for col in cat_col:
    assigning_missing_values(col)

le = LabelEncoder()

for col in cat_col:
    data[col] = le.fit_transform(data[col])


def data_split(entry, ratio):
    # np.random.seed(42)
    shuffled = np.random.permutation(len(entry))
    test_set_size = int(len(entry) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":

    # Read the data 
    # df = pd.read_csv('kidney_disease.csv')
    

    train, test = data_split(data, 0.25)


    X_train = train[['age', 'blood pressure', 'specific gravity', 'albumin', 'sugar',
        'red blood cells', ' pus cell', 'pus cell clumps', 'bacteria',
        'blood glucose random', 'blood urea', 'serum creatinine', 'sodium',
        'potassium', 'haemoglobin', 'packed cell volume',
        'white blood cell count', 'red blood cell count', 'ypertension',
        'diabetes mellitus', 'coronary artery disease', 'appetite',
        'pedal edema', 'anemia']].to_numpy()


    X_test = test[['age', 'blood pressure', 'specific gravity', 'albumin', 'sugar',
        'red blood cells', ' pus cell', 'pus cell clumps', 'bacteria',
        'blood glucose random', 'blood urea', 'serum creatinine', 'sodium',
        'potassium', 'haemoglobin', 'packed cell volume',
        'white blood cell count', 'red blood cell count', 'ypertension',
        'diabetes mellitus', 'coronary artery disease', 'appetite',
        'pedal edema', 'anemia']].to_numpy()
    
    Y_train = train[['class']].to_numpy().reshape(300,)
    Y_test = test[['class']].to_numpy().reshape(100,)


    clf = LogisticRegression(solver='lbfgs', max_iter=10000000)
    clf.fit(X_train, Y_train)

    file = open('model.pkl','wb')

    pickle.dump(clf, file)

    input_features = [5.0,	70.0,	1.025,	1.0,	0.0,	0,	1,	0,	0,	97.0,	
             56, 3.8, 111, 2.5, 11.2,
             34.0,	7200.0,	4.1,	0,	1,	0,	1,	0, 0]
    ckd_prob = clf.predict_proba([input_features])[0][1]

    print(ckd_prob)



