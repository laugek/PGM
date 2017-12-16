# Listing of attributes: credit to: https://archive.ics.uci.edu/ml/datasets/adult
# CLASS: >50K, <=50K.
# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

#################################################################################
##### Todo list:
#################################################################################
# bin: descrete variables also?

#################################################################################
#### import
#################################################################################
import time
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from pgmpy.models import NaiveBayes
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


start = time.time()
#################################################################################
#### data
#################################################################################

df = pd.read_csv("adult.csv", header=0, dtype=str) # read in data
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',\
 'marital-status', 'occupation', 'relationship', 'race', 'sex', \
 'capital-gain','capital-loss', 'hours-per-week', 'native-country', 'class']

###### DROPPING THESE COLOUMNS
df.drop('fnlwgt', axis=1, inplace=True) # is subjective
df.drop('native-country', axis=1, inplace=True)
df.drop('education', axis=1, inplace=True)
#df.drop('workclass', axis=1, inplace=True)
#df.drop('education-num', axis=1, inplace=True)
#df.drop('marital-status', axis=1, inplace=True)
#df.drop('occupation', axis=1, inplace=True)
df.drop('relationship', axis=1, inplace=True)
#df.drop('capital-gain', axis=1, inplace=True)
#df.drop('capital-loss', axis=1, inplace=True)
#df.drop('hours-per-week', axis=1, inplace=True)
#df.drop('race', axis=1, inplace=True)
#df.drop('age', axis=1, inplace=True)
#df.drop('sex', axis=1, inplace=True)

######## FIXING THE DATA
#df.drop(df.index[5000:], inplace=True) # use this line to reduce the dataset
print('Samples in total:', len(df))
df.replace(['?'], np.nan, inplace = True)
df.dropna(inplace=True)
print('Samples after removing missing varibles:', len(df))
#################################################################
### binning master
#################################################################
# age binning
bins = []
names=[]
feature = 'age'
df[feature] = pd.to_numeric(df[feature])
bins = [-math.inf, 21, 23, 27, 29, 35, 43, 54, 61, math.inf]
names = [0, 1, 2, 3, 4, 5, 6, 7, 8]
df[feature+'_bin'] = pd.cut(df[feature], bins, labels=names)
df.drop(feature, axis=1, inplace=True)

# # work hours binning
bins = []
names=[]
feature = 'hours-per-week'
df[feature] = pd.to_numeric(df[feature])
bins = [-math.inf, 34, 39, 41, 49, 65, math.inf]
names = [0, 1, 2, 3, 4, 5]
df[feature+'_bin'] = pd.cut(df[feature], bins, labels=names)
df.drop(feature, axis=1, inplace=True)

# capital-gain binning
bins = []
names=[]
feature = 'capital-gain'
df[feature] = pd.to_numeric(df[feature])
bins = [-math.inf, 0, 4101, 4386, 4687, 4865, 5060, 6418, 6849, math.inf]
names = [0,1,2,3,4,5,6,7,8]
df[feature+'_bin'] = pd.cut(df[feature], bins, labels=names)
df.drop(feature, axis=1, inplace=True)

# capital-loss binning
bins = []
names=[]
feature = 'capital-loss'
df[feature] = pd.to_numeric(df[feature])
bins = [-math.inf, 1504, 1564, 1816, 1876, 1977, 2206, 2377, 2559, math.inf]
names = [0,1,2,3,4,5,6,7,8]
df[feature+'_bin'] = pd.cut(df[feature], bins, labels=names)
df.drop(feature, axis=1, inplace=True)

# education-num binning
bins = []
names=[]
feature = 'education-num'
df[feature] = pd.to_numeric(df[feature])
bins = [-math.inf, 8, 9, 10, 12, 13, 14, math.inf]
names = [0,1,2,3,4,5,6]
df[feature+'_bin'] = pd.cut(df[feature], bins, labels=names)
df.drop(feature, axis=1, inplace=True)


# Print an example of 1 instance in the dataset
print("\nAn example of a person")
print(df.iloc[0])

# Split the data to test and train
test_size = 0.33
print("\nSplitting in to training and test data using: Test size = ", test_size)
data_train, data_test = train_test_split(df, test_size=test_size)
print("training data:", len(data_train))
print("test data:", len(data_test))


#################################################################################
##### Defining the model
#################################################################################
model = NaiveBayes()

# Learning CPDs using Maximum Likelihood Estimators
model.fit(data_train, 'class', estimator=MaximumLikelihoodEstimator)
# Print the CPDs learned
print("\n\n............Overview of our CPDs from the fit...........:")
for cpd in model.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)

print("\n\n............Overview of levels in variables...........:")
for col in df:
    print(col,":", len(df[col].unique()) )

#################################################################################
##### Using the model to query
#################################################################################
# Doing exact inference using Variable Elimination
model_infer = VariableElimination(model)
# Computing the probability of class given sex

# class = 0 is <= 50
# class = 1 is >50

# sex = 0 is female
# sex = 1 is male

# race = 2 is black
# race = 4 is white

# age = 1 is 21-23 yo
# age = 6 is 35-43 yo

# print('\nQuery: ')
# q = model_infer.query(variables=['age_bin'], evidence={'class': 0})
# print(q['age_bin'])
#
# print('\nQuery: ')
# q = model_infer.query(variables=['age_bin'], evidence={'class': 1})
# print(q['age_bin'])


print("\n\n............Here are some queries...............")
print('Query: Female')
q = model_infer.query(variables=['class'], evidence={'sex': 0})
print(q['class'])

print('Query: Male')
q = model_infer.query(variables=['class'], evidence={'sex': 1})
print(q['class'])

print('Query: White')
q = model_infer.query(variables=['class'], evidence={'race': 4})
print(q['class'])

print('Query: Black')
q = model_infer.query(variables=['class'], evidence={'race': 2})
print(q['class'])

print('Query: 21-23 years old')
q = model_infer.query(variables=['class'], evidence={'age_bin': 1})
print(q['class'])

print('Query: 35-43 years old')
q = model_infer.query(variables=['class'], evidence={'age_bin': 6})
print(q['class'])

print('Query: workhours 34-39')
q = model_infer.query(variables=['class'], evidence={'hours-per-week_bin': 1})
print(q['class'])

print('Query: workhours 49-65')
q = model_infer.query(variables=['class'], evidence={'hours-per-week_bin': 4})
print(q['class'])




#################################################################################
##### Evalutating the model by predicting
#################################################################################
# y_true = data_test['class'].copy()
# data_test.drop('class', axis=1, inplace=True)
# y_pred = model.predict(data_test)
#
# accuracy = accuracy_score(y_true, y_pred)
# print("\n\n\n\n\n\nAccuracy = ", accuracy)
# print("\n\nSince our data is skewed we should take a deeper look in to the results:")
# print(classification_report(y_true, y_pred))

print("\nEnd of code \n...o0o.... Fuck you Julien ...o0o...")
print("\nRuntime: ")
end = time.time()
print(round(end - start),"seconds")
