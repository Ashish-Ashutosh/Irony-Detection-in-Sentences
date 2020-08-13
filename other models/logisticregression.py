
'''
Logistic  regression with TFID and CountVectoriser
'''

import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression



df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')


df_x = df["TweetText"]
df_y = df["IndexLabel"]

cv = CountVectorizer()
#cv = TfidfVectorizer(min_df=1, stop_words='english')    #remove the stop words from the english language


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


x_traincv = cv.fit_transform(x_train.values.astype(str))
array = x_traincv.toarray()



#machine learning algorithm used to classify(Naive Bayes)
#mnb = MultinomialNB()
model = LogisticRegression()
#y_train = y_train.astype('int')    #converting type to int of training data of target class
print(y_train)
#creating the classifier
print(model.fit(x_traincv, y_train))



#extract tfid features to test data too

x_testcv=  cv.transform(x_test.values.astype(str))
#for checking predictions
predictions = model.predict(x_testcv)

#to look at how many predictions were right
actual_results= np.array(y_test)
print(actual_results)

count = 0
for i in range (len(predictions)):
    if predictions[i] == actual_results[i]:
        count = count+1

#count is the number of correct predictions
print(count)

#predictions is total number of predictions
print(len(predictions))





