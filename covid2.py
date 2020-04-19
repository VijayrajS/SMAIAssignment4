import numpy as np
import pandas as pd
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import rdkit
from rdkit import Chem 
from rdkit.Chem import rdMolDescriptors as Des

df= pd.read_csv('Datasets/COVID/train.csv', names=['smiles', 'aff'], skiprows = 1)

df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 

# df['mol'] = df['mol'].apply(lambda x: Chem.AddHs(x))
mdf = df.drop(columns=['aff'])
y = df['aff'].values

from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec

model = word2vec.Word2Vec.load('./model_300dim.pkl')
#Constructing sentences
mdf['sentence'] = mdf.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

#Extracting embeddings to a numpy.array
#Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
mdf['mol2vec'] = [DfVec(x) for x in sentences2vec(mdf['sentence'], model, unseen='UNK')]
x = np.array([x.vec for x in mdf['mol2vec']])

from sklearn import preprocessing
x_scaled = x

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
print(x_scaled.shape)
# train_df = pd.DataFrame(x_scaled)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

mae_a = []
mse_a = []
err_A = []

for i in range(0,20):

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=.2, random_state=i)

    reg = LinearRegression().fit(np.array(X_train), np.array(y_train))
    y_e = reg.predict(X_test)


    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mse = mean_squared_error(y_e,y_test)
    mae = mean_absolute_error(y_e,y_test)

    err = [abs((y_e[i]-y_test[i])/y_test[i]) for i in range(len(y_e))]
    e_m = sum(err)/len(err)
    mae_a.append(mae)
    mse_a.append(mse)
    err_A.append(e_m)

# print(mae_a)
# print(mse_a)
# print(err_A)

print(np.mean(mae_a),np.mean(mse_a),np.mean(err_A))
# print(train_df)
