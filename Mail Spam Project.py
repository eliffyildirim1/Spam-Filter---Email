#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import os
import email
import email.policy
from bs4 import BeautifulSoup
from sklearn.model_selection import cross_val_score, train_test_split
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

os.listdir('C:/Users/Elif/Desktop/archive/hamnspam/')


#  ### Data Exploration

# In[17]:


ham_filenames = [name for name in sorted(os.listdir('C:/Users/Elif/Desktop/archive/hamnspam/ham')) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir('C:/Users/Elif/Desktop/archive/hamnspam/spam')) if len(name) > 20]


# In[18]:


print('Amount of ham files:', len(ham_filenames))
print('Amount of spam files:', len(spam_filenames))    
print('Spam to Ham Ratio:',len(spam_filenames)/len(ham_filenames))


# In[21]:


def load_email(is_spam, filename):
    directory = "C:/Users/Elif/Desktop/archive/hamnspam/spam" if is_spam else "C:/Users/Elif/Desktop/archive/hamnspam/ham"
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
    
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    
    
testEmail = ham_emails[0]


# In[22]:


print('Header Field Names:',testEmail.keys())
print('\n\n')
print('Message Field Values:',testEmail.values())
print('\n\n')
print('Message Content:',testEmail.get_content())


# In[23]:


testEmailContent = testEmail.get_content()
type(testEmailContent)


# In[24]:


testEmail['Subject']


# In[25]:


print(spam_emails[6].get_content())


# ### Turning Emails into plaintext
# 

# In[26]:


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

ham_structure = structures_counter(ham_emails)
spam_structure = structures_counter(spam_emails)


# In[27]:


ham_structure.most_common()


# In[28]:


spam_structure.most_common()


# In[29]:


for email in spam_emails:
    if get_email_structure(email) == 'text/html':
        testEmail = email
        break

print(testEmail.get_content())


# In[30]:


def html_to_plain(email):
    try:
        soup = BeautifulSoup(email.get_content(), 'html.parser')
        return soup.text.replace('\n\n','')
    except:
        return "empty"

print(html_to_plain(testEmail))


# In[31]:


def email_to_plain(email):
    struct = get_email_structure(email)
    for part in email.walk():
        partContentType = part.get_content_type()
        if partContentType not in ['text/plain','text/html']:
            continue
        try:
            partContent = part.get_content()
        except: # in case of encoding issues
            partContent = str(part.get_payload())
        if partContentType == 'text/plain':
            return partContent
        else:
            return html_to_plain(part)

print(email_to_plain(ham_emails[42]))
print(email_to_plain(spam_emails[42]))


# In[32]:


import nltk

stemmer = nltk.PorterStemmer()
for word in ("Working", "Work", "Works", "Worked"):
        print(word, "=>", stemmer.stem(word))


# In[33]:



# - Strip email headers
# - Convert to lowercase
# - Remove punctuation
# - Replace urls with "URL"
# - Replace numbers with "NUMBER"
# - Perform Stemming (trim word endings with library)
class EmailToWords(BaseEstimator, TransformerMixin):
    def __init__(self, stripHeaders=True, lowercaseConversion = True, punctuationRemoval = True, 
                 urlReplacement = True, numberReplacement = True, stemming = True):
        self.stripHeaders = stripHeaders
        self.lowercaseConversion = lowercaseConversion
        self.punctuationRemoval = punctuationRemoval
        self.urlReplacement = urlReplacement
        #self.url_extractor = urlextract.URLExtract()
        self.numberReplacement = numberReplacement
        self.stemming = stemming
        self.stemmer = nltk.PorterStemmer()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_to_words = []
        for email in X:
            text = email_to_plain(email)
            if text is None:
                text = 'empty'
            if self.lowercaseConversion:
                text = text.lower()
                
            #if self.urlReplacement:
                #urls = self.url_extractor.find_urls(text)
                #for url in urls:
                #    text = text.replace(url, 'URL')   
                    
            if self.punctuationRemoval:
                text = text.replace('.','')
                text = text.replace(',','')
                text = text.replace('!','')
                text = text.replace('?','')
                
            word_counts = Counter(text.split())
            if self.stemming:
                stemmed_word_count = Counter()
                for word, count in word_counts.items():
                    stemmed_word = self.stemmer.stem(word)
                    stemmed_word_count[stemmed_word] += count
                word_counts = stemmed_word_count
            X_to_words.append(word_counts)
        return np.array(X_to_words)


# In[34]:


X_few = ham_emails[:3]
Xwordcounts = EmailToWords().fit_transform(X_few)
Xwordcounts


# In[35]:


class WordCountToVector(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_word_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_word_count[word] += min(count, 10)
        self.most_common = total_word_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(self.most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


# In[36]:


vocab_transformer = WordCountToVector(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(Xwordcounts)
X_few_vectors.toarray()


# In[37]:


vocab_transformer.vocabulary_


# In[38]:


from sklearn.pipeline import Pipeline

email_pipeline = Pipeline([
    ("Email to Words", EmailToWords()),
    ("Wordcount to Vector", WordCountToVector()),
])


# ### Training a Model
# 

# In[39]:


X = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


X_augmented_train = email_pipeline.fit_transform(X_train)


# In[41]:


log_clf = LogisticRegression(solver="liblinear", random_state=42)
score = cross_val_score(log_clf, X_augmented_train, y_train, cv=3)
score.mean()


# In[42]:


X_augmented_test = email_pipeline.transform(X_test)

log_clf = LogisticRegression(solver="liblinear", random_state=42)
log_clf.fit(X_augmented_train, y_train)

y_pred = log_clf.predict(X_augmented_test)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))


# In[ ]:




