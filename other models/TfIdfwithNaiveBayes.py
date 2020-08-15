#TfIdf(Term Frequency- Inverse Distribution Frequency) - sometimes we have words that are very common ( more weightage) and they dominate our results in the Machine Learning Algo.
#So TfIdf reduces this domination and also considers(give more weightage) words that are RARE but give more features to the Vector
'''


TFID with multinomial naive bayes
'''

import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB




#importing data from a CSV file (we can also define our own column name here
df = pd.read_csv('preprocessed_data.csv', encoding='utf-8')


#splitting the data into message (TwitterText) and result(Target Class) - (stored as data frames):
df_x = df["TweetText"]
df_y = df["IndexLabel"]


#using CoutVectorizer
cv = TfidfVectorizer(min_df=1, stop_words='english',ngram_range=(1, 2))    #remove the stop words from the english language

#Splitting into training and test data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)



#passing data to be transformed as input(Tweets)
x_traincv = cv.fit_transform(x_train.values.astype(str))     #fit is used to form features and transform is used to just change from text to numeric
#converting it to array
array = x_traincv.toarray()

print(array[0])

#to get the list of features (note that they are without stop words)
print(cv.get_feature_names())

#to cross check if the same values have been transformed
print(cv.inverse_transform(array[0]))


#machine learning algorithm used to classify(Naive Bayes)
mnb = MultinomialNB()
#y_train = y_train.astype('int')    #converting type to int of training data of target class
print(y_train)
#creating the classifier
print(mnb.fit(x_traincv, y_train))



#extract tfid features to test data too

x_testcv=  cv.transform(x_test.values.astype(str))
#for checking predictions
predictions = mnb.predict(x_testcv)

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





