import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import TfidfVectoriser

def preprocess(x):
    def deEmojify(inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')

    return deEmojify(x)
# v = "Happy #JohnMCainDay #JohnMcCainDayJune14th #JohnMcCainAmericanHero 🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸Also #FuckTrump"
print(preprocess(v))

df= pd.read_csv('Datasets/COVID/train.csv')

tfidf = TfidfVectorizer(min_df=1, analyzer='word',token_pattern=regex,
            ngram_range=(1, 3), sublinear_tf=True, stop_words = 'english')

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=.2, random_state=i)

tfidf_inter = tfidf.fit_transform(X_train)

tfidf.transform(X_test)

