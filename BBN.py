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

from pgmpy.models import BayesianModel
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
df.drop('fnlwgt', axis=1, inplace=True) # based on report
df.drop('native-country', axis=1, inplace=True) # based on report
df.drop('education', axis=1, inplace=True) # based on report
df.drop('relationship', axis=1, inplace=True) # because its same as mariatal?
#df.drop('workclass', axis=1, inplace=True)
#df.drop('education-num', axis=1, inplace=True)
#df.drop('marital-status', axis=1, inplace=True)
#df.drop('occupation', axis=1, inplace=True)
df.drop('capital-gain', axis=1, inplace=True)
df.drop('capital-loss', axis=1, inplace=True)
df.drop('hours-per-week', axis=1, inplace=True)
#df.drop('race', axis=1, inplace=True)
df.drop('age', axis=1, inplace=True)
#df.drop('sex', axis=1, inplace=True)

######## FIXING THE DATA
# use this line to reduce the dataset during testing only
# df.drop(df.index[1000:], inplace=True)

#this removes samples with incomplete data
print('Samples in total:', len(df))
df.replace(['?'], np.nan, inplace = True)
df.dropna(inplace=True)
print('Samples after removing missing varibles:', len(df))

#################################################################
### binning master
#################################################################
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
print("An example of a person")
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
# this model is the one from the phd, but modified:
#   native-country is removed
#   relationship is removed (merged with marital status)

model = BayesianModel([ ('race', 'marital-status'), \
                            ('marital-status', 'occupation'), \
                            ('marital-status', 'sex'), \
                            ('marital-status', 'education-num_bin'), \
                        ('race', 'sex'), \
                            ('sex', 'occupation'), \
                        ('race', 'class'), \
                            ('class', 'occupation'), \
                        ('workclass', 'education-num_bin'), \
                            ('education-num_bin', 'occupation'), \
                            ('education-num_bin', 'class'), \
                        ('workclass', 'occupation'), \
                         ] )

# model = BayesianModel([ ('race', 'marital-status'), \
#                             ('marital-status', 'occupation'), \
#                             ('marital-status', 'relationship'), \
#                                 ('relationship', 'education'), \
#                                 ('relationship', 'sex'), \
#                             ('marital-status', 'sex'), \
#                         ('race', 'sex'), \
#                             ('sex', 'occupation'), \
#                         ('race', 'native-country'), \
#                             ('native-country', 'class'), \
#                                 ('class', 'occupation'), \
#                         ('workclass', 'education'), \
#                             ('education', 'occupation'), \
#                             ('education', 'class'), \
#                         ('workclass', 'occupation'), \
#                          ] )

# Learning CPDs using Maximum Likelihood Estimators
print("Training the model...")
model.fit(data_train, estimator=MaximumLikelihoodEstimator)
print("Training finished...")
# Print the CPDs learned
print("\n\n............Overview of our CPDs from the fit...........:")
# print(model.get_cpds('class'))
# for cpd in model.get_cpds():
#     print("CPD of {variable}:".format(variable=cpd.variable))
#     print(cpd)

#################################################################################
##### Using the model to query
#################################################################################
# Doing exact inference using Variable Elimination
model_infer = VariableElimination(model)
# Computing the probability of class given sex
print("\n\n............Here are some queries...............")
print('Query: Female, black')
q = model_infer.query(variables=['class'], evidence={'sex': 0, 'race': 0})
print(q['class'])

print('Query: Male, white')
q = model_infer.query(variables=['class'], evidence={'sex': 1, 'race': 1})
print(q['class'])

#################################################################################
##### Evalutating the model by predicting
#################################################################################
# use this line to reduce the dataset during testing only
# data_test.drop(data_test.index[10:], inplace=True)

# here goes the real stuff
y_true = data_test['class'].copy()
data_test.drop('class', axis=1, inplace=True)
y_pred = model.predict(data_test)

accuracy = accuracy_score(y_true, y_pred)
print("\n\n\n\n\n\nAccuracy = ", accuracy)
print("\n\nSince our data is skewed we should take a deeper look in to the results:")
print(classification_report(y_true, y_pred))

print("\nEnd of code \n...o0o.... Fuck you Julien ...o0o...")
print("\nRuntime: ")
end = time.time()
print(round(end - start),"seconds")
