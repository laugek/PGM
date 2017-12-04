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
# bin and include: capital gain/loss
# bin: rest of the continous variables also?

#################################################################################
#### import
#################################################################################
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
df.drop('fnlwgt', axis=1, inplace=True) # where 1 is the axis number (0 for rows and 1 for columns.)
df.drop('workclass', axis=1, inplace=True)
df.drop('education', axis=1, inplace=True)
df.drop('education-num', axis=1, inplace=True)
df.drop('marital-status', axis=1, inplace=True)
df.drop('occupation', axis=1, inplace=True)
df.drop('relationship', axis=1, inplace=True)
df.drop('capital-gain', axis=1, inplace=True)
df.drop('capital-loss', axis=1, inplace=True)
#df.drop('hours-per-week', axis=1, inplace=True)
df.drop('native-country', axis=1, inplace=True)
df.drop('race', axis=1, inplace=True)
#df.drop('age', axis=1, inplace=True)
#df.drop('sex', axis=1, inplace=True)

######## FIXING THE DATA
df.drop(df.index[1000:], inplace=True) # use this line to reduce the dataset

# age binning
df['age'] = pd.to_numeric(df['age'])
age_bins = [0, 19, 30, 45, 200]
age_names = ['teenager', 'young', 'mid', 'old']
df['age_bin'] = pd.cut(df['age'], age_bins, labels=age_names)
df.drop('age', axis=1, inplace=True)

# work hours binning
df['hours-per-week'] = pd.to_numeric(df['hours-per-week'])
hours_bins = [0, 36, 44, 55, 200]
hours_names = ['few', 'normal', 'more', 'muchmore']
df['hours_bin'] = pd.cut(df['hours-per-week'], hours_bins, labels=hours_names)
df.drop('hours-per-week', axis=1, inplace=True)

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
model = NaiveBayes()

# Learning CPDs using Maximum Likelihood Estimators
model.fit(data_train, 'class', estimator=MaximumLikelihoodEstimator)
# Print the CPDs learned
print("\n\n............Overview of our CPDs from the fit...........:")
for cpd in model.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)

#################################################################################
##### Using the model to query
#################################################################################
# Doing exact inference using Variable Elimination
model_infer = VariableElimination(model)
# Computing the probability of class given sex
# print("\n\n............Here are some queries...............")
# q1 = model_infer.query(variables=['class'], evidence={'sex':0})
# print(q1['class'])


#################################################################################
##### Evalutating the model by predicting
#################################################################################
y_true = data_test['class'].copy()
data_test.drop('class', axis=1, inplace=True)
y_pred = model.predict(data_test)
#print(y_pred)
accuracy = accuracy_score(y_pred, y_true)
print("\n\n\n\n\n\nAccuracy = ", accuracy)
print("\nEnd of code \n...o0o.... Fuck you Julien ...o0o...")
print("\nRuntime: ")
end = time.time()
print(round(end - start),"seconds")
