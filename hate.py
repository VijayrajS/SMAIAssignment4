import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import re
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

def preprocess(x):
    def deEmojify(inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')

    x = x.replace('@', '')
    x = re.sub(r'^https?:\/\/.*[\r\n]*', '', x)
    x = x.lower()
    return deEmojify(x)
# v = "Happy #JohnMCainDay #JohnMcCainDayJune14th #JohnMcCainAmericanHero 🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸Also #FuckTrump"
# print(preprocess(v))

df= pd.read_csv('Datasets/HATE/train.csv')
# print(df)
df['text'] = df['text'].apply(lambda x: preprocess(x))
tfidf = TfidfVectorizer(min_df=1, analyzer='word', ngram_range=(1, 2), stop_words = 'english')

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

clf = SVC(kernel = 'linear', gamma='auto')
# clf = LinearSVC(random_state=0, tol=1e-5, C = 0.1)
# clf = RandomForestClassifier(max_depth=5, n_estimators=50, max_features=3)

# clf = AdaBoostClassifier(n_estimators=50)
# clf = LogisticRegression(random_state=0, C=10)
a_a = []
f_a = []

# for i in range(10):
#     X_train, X_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=.2, random_state=i)

#     X_t = tfidf.fit_transform(X_train)
#     clf.fit(X_t, y_train)

#     Data_test = tfidf.transform(X_test)
#     pred = clf.predict(Data_test)

#     from sklearn.metrics import f1_score, accuracy_score
#     a = accuracy_score(pred, y_test)
#     f = f1_score(pred, y_test)
#     a_a.append(a)
#     f_a.append(f)

X_t = tfidf.fit_transform(df['text'])
clf.fit(X_t, df['labels'])

df1= pd.read_csv('Datasets/HATE/test.csv')
df1['text'] = df1['text'].apply(lambda x: preprocess(x))
d_test = tfidf.transform(df1['text'])
pred = clf.predict(d_test)

print('labels')
print('\n'.join(str(u) for u in pred))

# print(np.mean(a_a),np.mean(f_a))
