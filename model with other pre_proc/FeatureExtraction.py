
'''
All combinations
'''
import itertools
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer




#to print out the entire matrix after vectorising it
#np.set_printoptions(threshold=np.nan)


#reading the training data
df = pd.read_csv('preprocessed_trainingdata.csv', encoding='utf-8')

#reading the test data
df1 = pd.read_csv('preprocessed_testdata.csv', encoding='utf-8')



#separating the target class from the data
df_x = df["TweetText"]
df_y = df["IndexLabel"]

df_x1 = df1["TweetText"]
df_y1 = df1["IndexLabel"]



#VECTORISER USED
cv = CountVectorizer()
#cv= DictVectorizer(sparse=False)
#cv = TfidfVectorizer(min_df=0, stop_words='english', ngram_range=(1, 2))    #remove the stop words from the english language


#x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

#to convert from list to dictionary because fit_tranform for DictVectorizer takes dictionary as parameter
#d = dict(itertools.zip_longest(*[iter(df_x)] * 2, fillvalue=""))
#x_traincv = cv.fit_transform(d)
x_traincv = cv.fit_transform(df_x.values.astype(str))
#x_traincv = cv.fit_transform(x_train.values.astype(str))
#array = x_traincv.toarray()
print(x_traincv)




#mnb = MultinomialNB()
#model = LinearSVC()
#model = tree.DecisionTreeClassifier()
model = LogisticRegression()
#y_train = y_train.astype('int')    #converting type to int of training data of target class
print(df_y)
#creating the classifier
print(model.fit(x_traincv, df_y))



#extract tfid features to test data too

x_testcv = cv.transform(df_x1.values.astype(str))
#for checking predictions
predictions = model.predict(x_testcv)

#to look at how many predictions were right
actual_results= np.array(df_y1)
print(actual_results)

count = 0
for i in range(len(predictions)):
    if predictions[i] == actual_results[i]:
        count = count+1

#count is the number of correct predictions
a=count
print(count)

#predictions is total number of predictions
b=len(predictions)
print(len(predictions))

accuracy = a/b

print(accuracy)







