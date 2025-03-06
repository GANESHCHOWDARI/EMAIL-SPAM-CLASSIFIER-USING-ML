import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load the data
data=pd.read_csv('./spam.csv')
# print(data.head()) #checking the first 5 rows of the data
# print(data.shape)
data.drop_duplicates(inplace=True) #dropping duplicates
# print(data.shape) #checking the shape of the data


#cheching for null values in the data
# print(data.isnull().sum())

data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

mess=data['Message']
cat=data['Category']

(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

cv=CountVectorizer(stop_words='english')
features=cv.fit_transform(mess_train)


#creating model

model=MultinomialNB()
model.fit(features, cat_train)


#testing the model
feature_test=cv.transform(mess_test)
# print(model.score(feature_test, cat_test)) #accuracy of the model


#predicting the category of the message

# message=cv.transform(["We are pleased to inform you that you have won the international lottery! Your winning amount is $1,000,000. To claim your prize, please provide your full name, address, and banking details."]).toarray()
# result=model.predict(message)
# print(result) #category of the message
def predict(message):
    input_message=cv.transform([message]).toarray()
    result=model.predict(input_message)
    return result
st.header('Spam Detection App')
user_input = st.text_input('Enter your message here:')
if st.button('Predict'):
    result = predict(user_input)
    st.write(result)