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
df['mol'] = df['mol'].apply(lambda x: Chem.AddHs(x))

df['num_of_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
df['num_of_bonds'] = df['mol'].apply(lambda x: x.GetNumBonds())
df['num_of_bonds_sq'] = df['mol'].apply(lambda x: x.GetNumBonds()**2)

# df['ringsar'] = df['mol'].apply(lambda x: Des.CalcNumAromaticRings(x))
df['rings_sq'] = df['mol'].apply(lambda x: Des.CalcNumAromaticRings(x)**2)

# df['num_of_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
df['n_h_a_sq'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms()**2)

def number_of_atoms(atom_list, df):
    for i in atom_list:
        df['num_of_{}_atoms'.format(i)] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))

number_of_atoms(['C','O','N'], df)
############################################

train_df = df.drop(columns=['smiles', 'mol', 'aff'])
y = df['aff'].values

from sklearn import preprocessing

x = train_df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

train_df = pd.DataFrame(x_scaled)

from sklearn.model_selection import train_test_split

mae_a = []
mse_a = []
err_A = []

for i in range(0,20):

    X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=.2, random_state=i)

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(np.array(X_train), np.array(y_train))
    y_e = reg.predict(X_test)

    # from sklearn import linear_model
    # reg = linear_model.BayesianRidge(n_iter = 1000)
    # reg.fit(np.array(X_train), np.array(y_train))
    # y_e = reg.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mse = mean_squared_error(y_e,y_test)
    mae = mean_absolute_error(y_e,y_test)

    err = [abs((y_e[i]-y_test[i])/y_test[i]) for i in range(len(y_e))]
    e_m = sum(err)/len(err)
    mae_a.append(mae)
    mse_a.append(mse)
    err_A.append(e_m)

print(mae_a)
print(mse_a)
print(err_A)

print(np.mean(mae_a),np.mean(mse_a),np.mean(err_A))
# print(train_df)
