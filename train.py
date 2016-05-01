import sys
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = 10, 8

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

train_data = pd.read_csv("translated_1000_train_1.csv")
test_data = pd.read_csv("translated_1000_train_2.csv")
stopwords_data = open("translated_stopwords_ru.txt").read().splitlines()

train_data_X = train_data.drop(['9'], 1)
train_data_Y = train_data['9']

test_data_X = test_data.drop(['9'], 1)
test_data_Y = test_data['9']

base_rate_train=0
for row in train_data_Y:
    if row == 1:
        base_rate_train+=1
print(base_rate_train)

binary_vectorizer = CountVectorizer(ngram_range=[1,3],stop_words=stopwords_data)
binary_vectorizer.fit(train_data['2'])

matrix = binary_vectorizer.transform(train_data['2'])

df = pd.DataFrame(matrix.toarray().transpose(), index = binary_vectorizer.get_feature_names())
df_tr = pd.DataFrame.transpose(df)

train_matrix = pd.concat([train_data_X, df_tr], axis=1)
test_matrix = pd.concat([test_data_X, df_tr], axis=1)

for i in range (3,5):
    binary_vectorizer.fit(train_data[str(i)])
    matrix = binary_vectorizer.transform(train_data[column_string])
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
naive_bayes = BernoulliNB()

tree.fit(train_matrix, train_data_Y)
logistic.fit(train_matrix, train_data_Y)
naive_bayes.fit(train_matrix, train_data_Y)

Y_test_probabilities_tree = tree.predict_proba(test_matrix)[:, 1]
Y_test_probabilities_logistic = logistic.predict_proba(test_matrix)[:, 1]
Y_test_probabilities_nb = naive_bayes.predict_proba(test_matrix)[:, 1]

fpr_tree, tpr_tree, thresholds_tree = roc_curve(test_data_Y, Y_test_probabilities_tree)
fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(test_data_Y, Y_test_probabilities_logistic)
fpr_nb, tpr_nb, thresholds_logistic = roc_curve(test_data_Y, Y_test_probabilities_nb)

print(cross_validation.cross_val_score(logistic, test_matrix, test_data_Y, scoring="roc_auc", cv=5))
print(cross_validation.cross_val_score(tree, test_matrix, test_data_Y, scoring="roc_auc", cv=5))
print(cross_validation.cross_val_score(naive_bayes, test_matrix, test_data_Y, scoring="roc_auc", cv=5))

auc_tree = roc_auc_score(test_data_Y, Y_test_probabilities_tree)
auc_logistic = roc_auc_score(test_data_Y, Y_test_probabilities_logistic)
auc_nb = roc_auc_score(test_data_Y, Y_test_probabilities_nb)

auc_tree = round(auc_tree, 2)
auc_logistic = round(auc_logistic, 2)
auc_nb = round(auc_nb, 2)

plt.plot(fpr_tree, tpr_tree, label="md=" + str(20) + ", AUC (Tree) =" + str(auc_tree))
plt.plot(fpr_logistic, tpr_logistic, label="md=" + str(20) + ", AUC (Logistic)=" + str(auc_logistic))
plt.plot(fpr_nb, tpr_nb, label="md=" + str(20) + ", AUC (Naive-bayes)=" + str(auc_nb))

#Y_predicted = logistic.predict(train_matrix)
#accuracy = accuracy_score(Y_predicted, Y)
#print("The accuracy is " + str(accuracy))

plt.plot([0,1.0], [0,1.0], 'k--', label="Random")
plt.legend(loc=2)

plt.show()
