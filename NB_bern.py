
# import data
import csv
import collections
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
#import numpy as np


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
# setting up some classes
#################################################################################

class Person():
    pass

    def setup(self, headers, info):
        for idx, head in enumerate(headers):
            if head[0] == " ":
                head = head[1:]
            try:
                dat = info[idx]
            except:
                dat = [" "]
            if dat[0] == " ":
                dat = dat[1:]
            setattr(Person, "a_"+head, dat)

    def binarize(self):
        # age
        if int(self.a_Age) > 45:
            setattr(Person, 'b_old', 1)
        else:
            setattr(Person, 'b_old', 0)
        # race
        if self.a_race == 'White':
            setattr(Person, 'b_race', 1)
        else:
            setattr(Person, 'b_race', 0)
        # workclass
        if self.a_workclass == 'Private':
            setattr(Person, 'b_private', 1)
        else:
            setattr(Person, 'b_private', 0)
        # education
        if self.a_education == 'Bachelors' or self.a_education == 'Masters' \
        or self.a_education == 'Doctorate' :
            setattr(Person, 'b_longEdu', 1)
        else:
            setattr(Person, 'b_longEdu', 0)
        # sex
        if self.a_sex == 'Male' :
            setattr(Person, 'b_sex', 1)
        else:
            setattr(Person, 'b_sex', 0)
        # class
        if self.a_class == '<=50K' :
            setattr(Person, 'b_rich', 1)
        else:
            setattr(Person, 'b_rich', 0)

    def dump(self):
        all = [attr for attr in dir(self) if not attr.startswith('__')]
        for attr in all:
            print("%s = %s" % (attr, getattr(self, attr)))

#################################################################################
# importing the data
#################################################################################

peops = []
x = []
y = []

original = open('adult.csv', newline='')
with original as in_file:
    data = list(csv.reader(in_file))
    headers = data[0]
    print("The header (first data row):")
    print(headers)
    print("\nAn example of a person:")
    print(data[1])
    N = len(data)
    #for i in range(1,1000):
    for i in range(1,N-1):
        temp = Person()
        temp.setup(headers, data[i])
        peops.append(temp)
        # This section makes it binary
        peops[i-1].binarize()
        # we take out the binary data
        binData = [peops[i-1].b_old, peops[i-1].b_race, peops[i-1].b_private, \
        peops[i-1].b_longEdu, peops[i-1].b_sex]
        x.append(binData)
        y.append(peops[i-1].b_rich)

print("\nAn example of a person in binary :")
print(x[0])

#################################################################################
# Split x and y into test and train
#################################################################################
test_size = 0.33
print("\nSplitting in to training and test data using: Test size = ", test_size)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size)
print("training data:", len(y_train))
print("test data:", len(y_test))

#################################################################################
# training the model
#################################################################################
clf = BernoulliNB()
clf.fit(x_train, y_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

#################################################################################
# testing our model
#################################################################################
c1 = 0
c0 = 0
for j in range(0,len(y_test)):
    #print("Prediction value:", clf.predict( [x_test[j]] )[0] )
    #print("Real value:", y[j])
    if y[j] == clf.predict( [x_test[j]] )[0]:
        c1 += 1
    else:
        c0 += 1
print("\nResults of the test:")
print("Correct predictions:", c1)
print("False predictions:", c0)
print("Accuracy = ", (c1-c0)/c1*100)
