import sys
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = 10, 8

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

train_data = pd.read_csv("translated_1000_train_1.csv")
test_data = pd.read_csv("translated_1000_train_2.csv")

train_data_X = train_data.drop(['9'], 1)
train_data_Y = train_data['9']

test_data_X = test_data.drop(['9'], 1)
test_data_Y = test_data['9']

base_rate_train=0
for row in train_data_Y:
    if row == 1:
        base_rate_train+=1
print("The base rate for the training data is " + str(base_rate_train/1000))

binary_vectorizer = CountVectorizer(binary=True)
binary_vectorizer.fit(train_data['2'])

matrix = binary_vectorizer.transform(train_data['2'])

df = pd.DataFrame(matrix.toarray().transpose(), index = binary_vectorizer.get_feature_names())

df_tr = pd.DataFrame.transpose(df)

train_matrix = pd.concat([train_data_X, df_tr], axis=1)

test_matrix = pd.concat([test_data_X, df_tr], axis=1)

for i in range (3,5):
    column_string = str(i)
    binary_vectorizer.fit(data[column_string])
    matrix = binary_vectorizer.transform(data[column_string])
    df = pd.DataFrame(matrix.toarray().transpose(), index = binary_vectorizer.get_feature_names())
    df_tr = pd.DataFrame.transpose(df)
    train_matrix = pd.concat([train_matrix, df_tr], axis=1)
    test_matrix = pd.concat([test_matrix, df_tr], axis=1)
    
cols_to_drop = [1,2,3,4,5,6,8]

for col in cols_to_drop:
    train_matrix = train_matrix.drop([str(col)],1)
    test_matrix = test_matrix.drop([str(col)],1)

tree = DecisionTreeClassifier(criterion="entropy", max_depth = 5)
logistic = LogisticRegression()

tree.fit(train_matrix, Y)
logistic.fit(train_matrix, Y)

Y_test_probabilities_tree = tree.predict_proba(test_matrix)[:, 1]
Y_test_probabilities_logistic = logistic.predict_proba(test_matrix)[:, 1]

fpr_tree, tpr_tree, thresholds_tree = roc_curve(test_data_Y, Y_test_probabilities_tree)
fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(test_data_Y, Y_test_probabilities_logistic)

aucs_logistic = cross_validation.cross_val_score(logistic, test_matrix, test_data_Y, scoring="roc_auc", cv=5)
aucs_tree = cross_validation.cross_val_score(tree, test_matrix, test_data_Y, scoring="roc_auc", cv=5)

print(str(round(np.mean(aucs_logistic), 3)))
print(str(round(np.mean(aucs_tree), 3)))

auc1 = roc_auc_score(B, Y_test_probabilities_tree)
auc2 = roc_auc_score(B, Y_test_probabilities_logistic)

auc1 = round(auc1, 2)
auc2 = round(auc2, 2)

plt.plot(fpr_tree, tpr_tree, label="md=" + str(20) + ", AUC=" + str(auc1))
plt.plot(fpr_logistic, tpr_logistic, label="md=" + str(20) + ", AUC=" + str(auc2))

#Y_predicted = logistic.predict(train_matrix)
#accuracy = accuracy_score(Y_predicted, Y)
#print("The accuracy is " + str(accuracy))

plt.plot([0,1.0], [0,1.0], 'k--', label="Random")
plt.legend(loc=2)

plt.show()
