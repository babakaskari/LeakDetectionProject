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

# def unlabeled():

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
# ##################################################### CORRELATION MATRIX
# print(dataset.columns.values)
# dataset2 = dataset.drop(["Leak Found"], axis=1)
# df = pd.DataFrame(dataset2, columns=['Date', 'ID', 'value_Lvl', 'value_Spr'])
# corrMatrix = df.corr()
# sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
# ##################################################### SPLIT THE DATASET
x_train = dataset.loc[dataset['Leak Found'].isna()]
x_train = x_train.drop(["Leak Found"], axis=1)
# x_train = x_train.sample(frac=1)
x_test = dataset.loc[dataset['Leak Found'].notna()]
y_test = x_test.loc[dataset['Leak Found'].notna(), ['Leak Found']]
# ##################################################### CORRELATION OF KNOWN LABELLED DATA
df = pd.DataFrame(x_test, columns=['Date', 'ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
# #####################################################
x_test = x_test.drop(["Leak Found"], axis=1)
# print("x_test shape is equal to :  ", x_test.shape)
# print("dataset features :  ", dataset.columns)
# ############################################# CREATING DUMMY_DATA
# x_centroid = np.array(x_test.iloc[[16, 17], ])
dummy_data = dataset.drop(['Leak Found'], axis=1)
print("Description  : \n ", dummy_data.describe())
# ############################################# TO TAKE THE SELECTED SAMPLE FOR OUR XTRAIN
# dummy_data = dummy_data.sample(frac=1)
# x_dummy = dummy_data[:54]
# x_train = x_dummy
# ############################################ SCALER NORMALIZATION   " TO BE MODIFIED LATER"

# scaler = MinMaxScaler()
# # fit using the train set
# scaler.fit(x_train)
# # transform the test test
# x_train = scaler.transform(x_train)
# # build the scaler model
# scaler = Normalizer()
#
# # fit using the train set
# scaler.fit(x_train)
# # transform the test test
# x_train = scaler.transform(x_train)
# plt.show()
########################################### APPLYING GUASSRANK NORMALIZATION
"""
x_cols = x_train.columns[:]
x = x_train[x_cols]

s = GaussRankScaler()
x_ = s.fit_transform( x )
assert x_.shape == x.shape
x_train[x_cols] = x_
"""
############################################### standard scaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(data_scaled)
x_train.to_csv('Unlabeled.csv')
print("x_train description : \n", x_train.describe())
########################################## TO REPRESENT OUR DATASET, ALL COLUMNS IN MATRIX FORM
x_train = pd.DataFrame(x_train)

data_dict = {
    "x_train": x_train,
    "x_test": x_test,
}

    # return data_dict
