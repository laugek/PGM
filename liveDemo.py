#################################################################################
#### import
#################################################################################
import time
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from pgmpy.models import NaiveBayes
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


pd.options.mode.chained_assignment = None  # default='warn'

#################################################################################
#### data
#################################################################################
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('\n\n\n......................................................')
print('......................................................')
print("...................LIVE DEMO..........................")
print("............NAIVE BAYES CLASSIFIER....................")
print('......................................................')
print('......................................................\n\n')
input("\n ")

print("\n\n............Load data and perform binning...........:\n")
df = pd.read_csv("adult.csv", header=0, dtype=str) # read in data
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',\
 'marital-status', 'occupation', 'relationship', 'race', 'sex', \
 'capital-gain','capital-loss', 'hours-per-week', 'native-country', 'class']

###### DROPPING THESE COLOUMNS
df.drop('fnlwgt', axis=1, inplace=True) # is subjective

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
bins = [-math.inf, 34, 39,  41, 49, 65, math.inf]
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
# print("\n\n............An example of a person...........:\n")
# print(df.iloc[0])
# input("\n ")

# Split the data to test and train
print("Data set size:", len(df))
print("\n\n............Splitting the data in test and train...........:\n")
test_size = 0.33
print("Test size = ", test_size)
data_train, data_test = train_test_split(df, test_size=test_size)
print("training data:", len(data_train))
print("test data:", len(data_test))
input("\n ")


#################################################################################
##### Defining the model
#################################################################################
model = NaiveBayes()

# Learning CPDs using Maximum Likelihood Estimators
model.fit(data_train, 'class', estimator=MaximumLikelihoodEstimator)
# Print the CPDs learned
print("\n\n............Selected CPDs from the fit...........:\n")
print('CPD: class (parent of all nodes)')
print(model.get_cpds('class'))
input("\n ")
print('\nCPD: sex')
print(model.get_cpds('sex'))
# print(model.get_cpds('race'))

input("\n ")

# print("\n\n............Overview of levels in variables...........:\n")
# for col in df:
#     print(col,":", len(df[col].unique()) )
#
# input("\n ")

#################################################################################
##### Using the model to query
#################################################################################
# Doing exact inference using Variable Elimination
model_infer = VariableElimination(model)
# Computing the probability of class given sex
print("\n\n............Here are some queries...............:\n")
print('We use variable elimination (exact inference)\n\n')
print('\nQuery: Female, 16-21 years old')
q = model_infer.query(variables=['class'], evidence={'sex': 0, 'age_bin': 0})
print(q['class'])
input("\n ")
print('\nQuery: Male, 35-43 years old')
q = model_infer.query(variables=['class'], evidence={'sex': 1, 'age_bin': 6})
print(q['class'])

input("\n ")

#################################################################################
##### Evalutating the model by predicting
#################################################################################
print("\n\n............Now lets try and predict 10 peoples income..............: \n")
data_test.drop(data_test.index[10:], inplace=True) # use this line to reduce the dataset
y_true = data_test['class'].copy()
data_test.drop('class', axis=1, inplace=True)
y_pred = model.predict(data_test)
print('True class:\n', y_true)
input("")
print('\nPredicted class:\n', y_pred)
accuracy = accuracy_score(y_true, y_pred)
input("")
print("\n\nAccuracy = ", accuracy)

# input("\n")
#
# print("\n\nFurther evaluation of the network:\n")
# print(classification_report(y_true, y_pred))

print("\n ...ooo000 End of live demo 000ooo...\n\n")
