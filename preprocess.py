import pandas as pd
import string #library that contains punctuation
import re
from nltk.stem import WordNetLemmatizer
import numpy as np

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#defining function for tokenization
def tokenization(text):
    tokens = re.split(' ',text)
    return tokens

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def back_to_caption(msg_lemmatized):
    return " ".join(msg_lemmatized)

data = pd.read_csv("archive/captions.txt", sep=",") # load the captions.txt file from flickr8k dataset
data = data.sample(frac=1).reset_index(drop=True) # to shuffle

#storing the puntuation free text
data['clean_msg']= data['caption'].apply(lambda x:remove_punctuation(x))
data['msg_lower']= data['clean_msg'].apply(lambda x: x.lower())

#applying function to the column
data['msg_tokenized']= data['msg_lower'].apply(lambda x: tokenization(x))

#defining the object for Lemmatization, then lemmatize
wordnet_lemmatizer = WordNetLemmatizer()
data['msg_lemmatized']=data['msg_tokenized'].apply(lambda x:lemmatizer(x))

# batching based on length of caption
data['length']=data['msg_lemmatized'].apply(lambda x:len(x))
data = data[data.length>3]
data = data.sort_values('length').reset_index()

# put processed captions back into original column
data['caption']=data['msg_lemmatized'].apply(lambda x:back_to_caption(x))

# wrapping up
del data['clean_msg']
del data['msg_lower']
del data['msg_tokenized']
del data['msg_lemmatized']
del data['length']

# random, but sorted split indices
permutated_indices = np.random.permutation(data.shape[0])
train_indices, test_indices = np.sort(permutated_indices[:40000]), np.sort(permutated_indices[40000:])

# writing pre-processed annotation files:
# full set
data.caption.to_csv("archive/ann_caps.txt", sep='\t', index = False)
data.image.to_csv("archive/img_dirs.txt", sep='\t', index = False)
data.to_csv("archive/annotations.txt", sep='\t', index = False)

# training split
data.caption[train_indices].to_csv("archive/ann_caps_train.txt", sep='\t', index = False)
data.image[train_indices].to_csv("archive/img_dirs_train.txt", sep='\t', index = False)
data.iloc[train_indices].to_csv("archive/annotations_train.txt", sep='\t', index = False)

# validation split
data.caption[test_indices].to_csv("archive/ann_caps_val.txt", sep='\t', index = False)
data.image[test_indices].to_csv("archive/img_dirs_val.txt", sep='\t', index = False)
data.iloc[test_indices].to_csv("archive/annotations_val.txt", sep='\t', index = False)

# END