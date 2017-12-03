# Listing of attributes: credit to: https://archive.ics.uci.edu/ml/datasets/adult
#
# >50K, <=50K.
#
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
##### problmes:
#################################################################################
# cannot query in a naive bayes network. only works if evidence is parent

#################################################################################
#### import
#################################################################################
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pgmpy.models import BayesianModel
from pgmpy.models import NaiveBayes
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination

#################################################################################
#### data
#################################################################################
df = pd.read_csv("adult.csv", header=0, dtype=str) # read in data
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',\
 'marital-status', 'occupation', 'relationship', 'race', 'sex', \
 'capital-gain','capital-loss', 'hours-per-week', 'native-country', 'class']

# DROPPING THESE COLOUMNS
df.drop('fnlwgt', axis=1, inplace=True) # where 1 is the axis number (0 for rows and 1 for columns.)
df.drop('workclass', axis=1, inplace=True)
df.drop('education', axis=1, inplace=True)
df.drop('education-num', axis=1, inplace=True)
df.drop('marital-status', axis=1, inplace=True)
df.drop('occupation', axis=1, inplace=True)
df.drop('relationship', axis=1, inplace=True)
df.drop('capital-gain', axis=1, inplace=True)
df.drop('capital-loss', axis=1, inplace=True)
df.drop('hours-per-week', axis=1, inplace=True)
df.drop('native-country', axis=1, inplace=True)

df.drop('race', axis=1, inplace=True)

### FIXING THE DATA
df2 = df[0:2000]
#df2['race'].replace([' Black', ' White'], [1,2], inplace =True)
df2['sex'].replace([' Female', ' Male'], [0,1], inplace =True)
df2['class'].replace([' <=50K', ' >50K'], [0,1], inplace =True)
## age
df2['age'] = pd.to_numeric(df2['age'])
age_bins = [0, 19, 30, 45, 200]
age_names = ['teenager', 'young', 'mid', 'old']
df2['age_bin'] = pd.cut(df2['age'], age_bins, labels=age_names)
df2.drop('age', axis=1, inplace=True)


print("An example of a person")
print(df2.iloc[0])

test_size = 0.2
print("\nSplitting in to training and test data using: Test size = ", test_size)
data_train, data_test = train_test_split(df2, test_size=test_size)
print("training data:", len(data_train))
print("test data:", len(data_test))


#################################################################################
##### Defining the model
#################################################################################
model = BayesianModel([('age_bin', 'class'),('sex', 'class')])
#model = NaiveBayes([('class', 'age_bin'), ('class', 'sex')])
#model = BayesianModel([('sex', 'class')])
#model = BayesianModel([('class', 'sex')])

# Learing CPDs using Maximum Likelihood Estimators
model.fit(data_train, estimator=MaximumLikelihoodEstimator)

### independencies of network
print("independencies")
print(model.local_independencies('sex'))
#print(model.get_independencies())
#print(model.get_cpds('class'))

#################################################################################
##### using the model
#################################################################################
# Doing exact inference using Variable Elimination
model_infer = VariableElimination(model)
# Computing the probability of class given sex
q1 = model_infer.query(variables=['class'], evidence={'sex':0})
print(q1['class'])

q2 = model_infer.query(variables=['class'], evidence={'sex':1})
print(q2['class'])

q2 = model_infer.query(variables=['class'], evidence={'sex':1, 'age_bin':0})
print(q2['class'])

q2 = model_infer.query(variables=['class'], evidence={'sex':0, 'age_bin':2})
print(q2['class'])

#################################################################################
##### evalutating the model
#################################################################################

y_true = data_test['class']

data_test.drop('class', axis=1, inplace=True)

y_pred = model.predict(data_test)
#print(y_pred)

accuracy = accuracy_score(y_pred, y_true)
print("\n\n\n\n\n\nAccuracy = ", accuracy)
print("\nEnd of code \nFuck you Julien")
