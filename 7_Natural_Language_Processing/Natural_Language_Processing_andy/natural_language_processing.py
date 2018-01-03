# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset

dataset = pd.read_csv("./Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# Clean

corpus = []

import re
from nltk.stem.porter import PorterStemmer
#nltk.download("stopwords")
from nltk.corpus import stopwords

for i in range(0, len(dataset)):
        
    review = re.sub("[^a-zA-Z]", " ", dataset['Review'][i])
    review = review.lower()
    review = review.split()
        
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    
    corpus.append(review)

# Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # ignore very rare words like "steve"
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


#
# Text classification
#

# split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state = 42)

# scale
# Not needed here, since 0,1 output

# train
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state=42)
rf.fit(X_train, y_train)

nb = GaussianNB()
nb.fit(X_train, y_train)

# predict
y_pred_rf = rf.predict(X_test)
y_pred_nb = nb.predict(X_test)

# accuracy
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_nb = confusion_matrix(y_test, y_pred_nb)

acc_rf = sum(cm_rf.diagonal())/sum(sum(cm_rf))
acc_nb = sum(cm_nb.diagonal())/sum(sum(cm_nb))

print("Accuracy of Random forest classifier vs. Naive Bayes classifier is... "+str(acc_rf)+" : "+str(acc_nb))
print("\nCorresponding confusion matrices...")
print("Random Forest:")
print(str(cm_rf))
print("Naive Bayes:")
print(str(cm_nb))