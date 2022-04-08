import numpy as np
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import streamlit as st
import re

#loading the model and vectorizer
classifier = pickle.load(open('C:/Users/John Dave/anaconda3/envs/MultinomialSSF/Deployment/model.pkl','rb'))
cv = pickle.load(open('C:/Users/John Dave/anaconda3/envs/MultinomialSSF/Deployment/cvectorizer.pkl','rb'))
#creating a function for prediction

st.title("Multinomial as SMS Spam Filtering Web App")
user = st.text_input('Please input some words to classify:')
input_user = [user]
bow = []
def spam_prediction(input_data):
    for i in range(len(input_user)):
        reg = re.sub('[^a-zA-Z]',' ', input_user[i]) 
        lowercase = reg.lower()
        clean_words = [PorterStemmer().stem(word) for word in lowercase.split() if not word in stopwords.words('english')]
        join = ' '.join(clean_words)
        bow.append(join) 
    vec = cv.transform(bow).toarray()
    result = classifier.predict(vec)
    if result == 1:
       return "Message type is Spam!"
    else:
       return "Message type is Ham!"
   
def main():
    classify =''
    if st.button('Classify'):
        classify = spam_prediction(user)
        
    st.text_area('You input:',user)
    st.text(classify) 

if __name__ == '__main__':
    main()
    
    