import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from gaussrank import *
import warnings
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')


# def just_labeled():
df = pd.read_csv("../dataset/Acoustic Logger Data.csv")
df1 = df.loc[df["LvlSpr"] == "Lvl"]
df3 = df.loc[df["LvlSpr"] == "Spr"]
df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
              var_name='Date')
df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
              var_name='Date')
df5 = pd.merge(df2, df4, on=['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
df6 = df5.drop(['LvlSpr_Lvl', 'LvlSpr_Spr'], axis=1).dropna()
df6['Date'] = pd.to_datetime(df6['Date'], format='%d-%b')
df6['Date'] = df6['Date'].dt.strftime('%d-%m')

df7 = pd.read_csv("../dataset/Leak Alarm Results.csv")
df7['Date Visited'] = pd.to_datetime(df7['Date Visited'], format='%d/%m/%Y')
df7['Date Visited'] = df7['Date Visited'].dt.strftime('%d-%m')
df7 = df7.rename(columns={'Date Visited': 'Date'})

df8 = pd.merge(df6, df7, on=['ID', 'Date'], how='left')
df8 = df8.sort_values(['Leak Alarm', 'Leak Found']).reset_index(drop=True)
# df8["Leak Alarm"] = df8["Leak Alarm"].fillna(-1)
# df8["Leak Found"] = df8["Leak Found"].fillna(-1)
dataset = df8

# ##################################################### Delete these row indexes from dataFrame
indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
dataset.drop(indexNames, index=None, inplace=True)
dataset.reset_index(drop=True, inplace=True)
# ##################################################### DROPPING LEAK ALARM & LEAK FOUND
dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
# dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
dataset = dataset.drop(['Leak Alarm'], axis=1)

# ############################################################ Convert Date categorical to numerical
# dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
date_encoder = preprocessing.LabelEncoder()
date_encoder.fit(dataset['Date'])
# print(list(date_encoder.classes_))
dataset['Date'] = date_encoder.transform(dataset['Date'])
# print(dataset.to_string(max_rows=200))
print("Number of null values in dataset :\n", dataset.isna().sum())
# ##################################################### SPLIT THE DATASET
x_labeled_data = dataset.loc[dataset['Leak Found'].notna()]

y_labeled_data = x_labeled_data["Leak Found"]
x_labeled_data = x_labeled_data.drop(["Leak Found"], axis=1)

# ############################################## standard scaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(x_labeled_data)
x_train = pd.DataFrame(data_scaled)
# print("x_train after normalization : ", x_train.head())
# print("x_train description after normalization: ", x_labeled_data.describe())

# ############################################################
# x_train = x_train.sample(frac=1)
print('JUST 53 NORMALIZED (WITH SCALER) LABELED DATA: \n ',x_train)
x_train.to_csv('JustLabeled.csv')
x_unlabeled_data = dataset.loc[dataset['Leak Found'].isna()]
y_unlabeled_data = x_unlabeled_data.drop(["Leak Found"], axis=1)

print("ID in just_labeld dtaaset : ", x_labeled_data["ID"].nunique())
print("ID in just_labeld dtaaset : ", x_labeled_data["ID"].unique())
print("number of occurence : ", x_labeled_data['ID'].value_counts())

x_train, x_test, y_train, y_test = train_test_split(x_labeled_data,
                                                    y_labeled_data,
                                                    test_size=0.2,
                                                    random_state=42)
x_train, x_cv, y_train, y_cv = train_test_split(x_train,
                                                y_train,
                                                stratify=y_train,
                                                test_size=0.2)
# print('X_TRAIN: \n',x_train)
# print('Y_TRAIN: \n',y_train)
# print('X_TEST: \n',x_test)
# print('Y_TEST: \n',y_test)

data_dict = {
    "x_train": x_train,
    "y_train": y_train,
    "x_test": x_test,
    "y_test": y_test,
    "x_cv": x_cv,
    "y_cv": y_cv

}

    # return data_dict

