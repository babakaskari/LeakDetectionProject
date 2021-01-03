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

# def labeled():
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
df8["Leak Found"] = df8["Leak Found"].fillna(0)

dataset = df8

indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
# Delete these row indexes from dataFrame
dataset.drop(indexNames, index=None, inplace=True)
dataset.reset_index(inplace=True)
dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
# dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
dataset1 = dataset
dataset = dataset1.drop(['Leak Alarm'], axis=1)

# ############################################################ Convert Date categorical to numerical
# dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
date_encoder = preprocessing.LabelEncoder()
date_encoder.fit(dataset['Date'])
# print(list(date_encoder.classes_))
dataset['Date'] = date_encoder.transform(dataset['Date'])
# print(dataset.to_string(max_rows=200))
dataset = dataset.drop_duplicates()
print(" dataset description : \n", dataset.describe())
# ##############################################
dataset = dataset.drop(['index'], axis=1)

# corrolation matrix
# print(dataset.columns.values)
df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr'])
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()

# dataset = dataset.loc[:80]
# dataset = dataset.sample(frac=1)

print("Number of null values in dataset : \n", dataset.isna().sum())
print("datase shape before duplicate : ", dataset.shape)

leak_found = dataset.drop(['ID', 'Date', 'value_Lvl', 'value_Spr'], axis=1)
dataset2 = dataset.drop(['Leak Found'], axis=1)
dataset5 = dataset2.drop_duplicates()

print("datase shape after duplicate : ", dataset5.shape)
print("ID in whole dtaaset : ", dataset2["ID"].nunique())
print("ID in just_labeld dtaaset : ", dataset2["ID"].unique())
print("number of occurence : ", dataset2['ID'].value_counts())
print('THE MAIN DATASET\n',dataset2)
# print("leak found shape : ", leak_found.shape)
# print("dataset2 shape : ", dataset2.shape)
# ########################################## APPLYING GUASSRANK NORMALIZATION

# x_cols = dataset2.columns[:]
# x = dataset2[x_cols]
#
# s = GaussRankScaler()
# x_ = s.fit_transform(x)
# assert x_.shape == x.shape
# dataset2[x_cols] = x_
# print("GaussRankScaler dataset description :\n ", dataset2.describe())

# ############################################## standard scaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(dataset2)
dataset2 = pd.DataFrame(data_scaled)
dataset2.to_csv('Labeled.csv')
print("StandardScaler NORMALIZATION dataset description :\n ", dataset2.describe())

# ##############################################

x_train, x_test, y_train, y_test = train_test_split(dataset2,
                                                    leak_found,
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
    "y_cv": y_cv,


}

    # return data_dict

