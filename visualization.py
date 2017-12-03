
##################################################################
### Import.
##################################################################
import pandas as pd
import matplotlib.pyplot as plt

##################################################################
### Read in the data.
##################################################################
df = pd.read_csv("adult.csv", header=0, dtype=str) # read in data
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',\
 'marital-status', 'occupation', 'relationship', 'race', 'sex', \
 'capital-gain','capital-loss', 'hours-per-week', 'native-country', 'class']
print("An example of a person")
print(df.iloc[0])
# drop a couloumn fx the fnlwgt because i dont know what it is?
df.drop('fnlwgt', axis=1, inplace=True) # where 1 is the axis number (0 for rows and 1 for columns.)

#################################################################
### plot
#################################################################
autoplotfeatures = ['workclass', 'education', 'education-num',\
 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'class']
for feature in autoplotfeatures:
    count = pd.value_counts(df[feature].values)
    fig=plt.figure()
    count.plot.bar()
    plt.title(feature)
    plt.ylabel('#people')
    plt.show()

##################################################################
### other featurs
### these need some re-arranging of the x-axis and maybe bin together
### but i couldn't figure out how to get it done
##################################################################
feature = 'age'
count = pd.value_counts(df[feature].values)
fig=plt.figure()
count.plot.bar()
plt.title(feature)
plt.ylabel('#people')
plt.show()

feature = 'capital-gain'
count = pd.value_counts(df[feature].values)
fig=plt.figure()
count.plot.bar()
plt.title(feature)
plt.ylabel('#people')
plt.show()

feature = 'capital-loss'
count = pd.value_counts(df[feature].values)
fig=plt.figure()
count.plot.bar()
plt.title(feature)
plt.ylabel('#people')
plt.show()

feature = 'hours-per-week'
count = pd.value_counts(df[feature].values)
fig=plt.figure()
count.plot.bar()
plt.title(feature)
plt.ylabel('#people')
plt.show()

feature = 'native-country'
count = pd.value_counts(df[feature].values)
fig=plt.figure()
count.plot.bar()
plt.title(feature)
plt.ylabel('#people')
plt.show()
