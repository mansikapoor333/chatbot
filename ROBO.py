#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
#nltk.download('stopwords')
import numpy as np
import random
import string
#convert raw documents to a matric of TF-IDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer
#find similarities between words entered by the user and words in the corpus
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


f = open('/Users/mansikapoor/Downloads/emotion.txt', 'r', errors = 'ignore')
raw = f.read()
raw = raw.lower()
#nltk.download('punkt')
#nltk.download('wordnet')



# In[3]:


# converts to list of sentences
sent_tokens = nltk.sent_tokenize(raw)

#converts to list of words
word_tokens = nltk.sent_tokenize(raw)

#sent_tokens[:2]
word_tokens[:2]


# In[4]:


#LemTokens which will take as input the tokens and return normalized tokens.

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[5]:


# when you greet a chatbot
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[6]:


#using document similarity to generate response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if (req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response=robo_response+sent_tokens[idx]
        return robo_response
    


# In[ ]:


#what the bot says while starting and ending conversation depending on the user's input
flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




