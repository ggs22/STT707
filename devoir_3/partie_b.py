'''
STT707 - Devoir 3 partie B
Gabriel Gibeau Sanche - gibg2501
12 avril 2021
'''

import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

data = pd.DataFrame({'hote': [160, 35, 700, 961, 572, 441, 783, 65, 77, 741],
                     'habi': [28, 34, 354, 471, 537, 404, 1114, 43, 60, 332],
                     'prop': [0, 1, 229, 633, 279, 166, 387, 21, 189, 327],
                     'pare': [321, 178, 959, 1580, 1689, 1079, 4052, 294, 839, 1789],
                     'amis': [36, 8, 185, 305, 206, 178, 497, 79, 53, 311],
                     'teca': [141, 0,  292, 360, 748, 434, 1464, 57, 124, 236],
                     'viva': [45, 4,119, 162, 155, 178, 525, 18, 28, 102],
                     'dive': [65, 0, 140, 148, 112, 92, 387, 6, 53, 102]},
                    index=['Agri', 'Sala', 'Patr', 'CadS', 'CadM', 'Empl', 'Ouvr', 'Pers', 'Autr', 'NAct'])


#%%

# Quick version using 'prince' package
mca = prince.MCA()
fit_data = mca.fit(data)
ax = mca.plot_coordinates(
     X=data,
     ax=None,
     figsize=(6, 6),
     show_row_points=True,
     row_points_size=10,
     show_row_labels=True,
     show_column_points=True,
     column_points_size=30,
     show_column_labels=False,
     legend_n_cols=1
     )

plt.show()

#%%

# Long version with all calculation steps
data_total = data.copy()
# add new columns with row totals
data_total['Total'] = data_total.sum(axis=1)
# add new row with columns total, as well as "total total" (i.e. sum of all modalities)
data_total = data_total.append(pd.Series(data=data_total.sum(axis=0), name='Total'), ignore_index=False)

print(data_total)

# Get to total number of modalities
k = data_total.iloc[-1:, -1:].values[0][0]

# Calculate row & columns profiles
row_weights = np.divide(data_total.iloc[:-1:, -1:], k)
colum_weights = np.divide(data_total.iloc[-1:, :-1:], k)

row_profils = np.divide(data_total.iloc[:-1:, :-1:], np.array(data_total.iloc[:-1:, -1:]), axis=1)
column_profils = np.divide(data_total.iloc[:-1:, :-1:], np.array(data_total.iloc[-1:, :-1:]), axis=0)

# Create our metric
Di = np.diag(row_weights.values.T[0])
Dj = np.diag(colum_weights.values[0])

Ai = np.linalg.inv(Dj)
Aj = np.linalg.inv(Di)

# Get the unit gravity centers
gi = np.dot(Dj, np.ones(Dj.shape[0]))
gi = gi/np.linalg.norm(gi)

gj = np.dot(Di, np.ones(Di.shape[0]))
gj = gj/np.linalg.norm(gj)

# Create our points clouds
X = 1/k * np.dot(np.linalg.inv(Di), data.values)
Y = 1/k * np.dot(np.linalg.inv(Dj), data.values.T)

# Calculate the covariance matrices for our clouds
Vi = np.linalg.multi_dot((X.T, Di, X))
Vj = np.linalg.multi_dot((Y.T, Dj, Y))

ViAi = np.dot(Vi, Ai)
VjAj = np.dot(Vj, Aj)
eig_valsi, eig_vecsi = np.linalg.eig(ViAi)
eig_valsj, eig_vecsj = np.linalg.eig(VjAj)

# rounding error cause complexe number with all imaginary parts to 0, to avoid warning
# lets re-cast the real part only
eig_vecsj = eig_vecsj.real
eig_valsj = eig_valsj.real

# for some reason, some eigen values are permutated between
# eig_valsi and eig_valsj. It shouldn't so we manually replace the
# eigen values in eig_valsj, and the corresponding vectors in
# eig_valsj
swap = eig_valsj[5].copy()
eig_valsj[5] = eig_valsj[7].copy()
eig_valsj[7] = swap.copy()

swap = eig_vecsj[:, 5].copy()
eig_vecsj[:, 5] = eig_vecsj[:, 7].copy()
eig_vecsj[:, 7] = swap.copy()

del swap

# Lets get rid of floating point numbers imprecision
eig_valsj = np.round(abs(eig_valsj), 10)
eig_valsi = np.round(abs(eig_valsi), 10)

# sanity check
assert np.allclose(eig_vecsi[:, 0], gi)
assert np.allclose(eig_vecsj[:, 0], gj)
assert np.allclose(eig_valsi, eig_valsj[0:eig_valsi.shape[0]])

als = list()
for eig_vec in eig_vecsi.T:
    # als.append(np.dot(Ai, eig_vec)/np.linalg.norm(np.dot(Ai, eig_vec)))
    als.append(np.dot(Ai, eig_vec))
als = np.array(als)
# als = als.T

bls = list()
for eig_vec in eig_vecsj.T:
    bls.append(np.dot(Aj, eig_vec)/np.linalg.norm(np.dot(Aj, eig_vec)))
    # bls.append(np.dot(Aj, eig_vec))
bls = np.array(bls)
# bls = bls.T

# Sanity check A_i V_i a^l = lambda_l a^l
for i in range(0, als.shape[0]):
    assert np.allclose(np.linalg.multi_dot((Ai, Vi, als[i])), np.dot(eig_valsi[i], als[i]))
# Sanity check A_j V_j b^l = mu_l b^l
for i in range(0, bls.shape[0]):
    assert np.allclose(np.linalg.multi_dot((Aj, Vj, bls[i])), np.dot(eig_valsj[i], bls[i]))

xi = list()
for ix, al in enumerate(als):
    xi.append(np.dot(X, al))
    print(f'eig val: {eig_valsi[ix]}')
    print(f'squared norm: {np.power(np.linalg.norm(xi[ix]), 2)}')
    print(f'ratio: {eig_valsi[ix] / np.power(np.linalg.norm(xi[ix]), 2)}')
    print()

xi = np.array(xi)

xi2 = np.dot(X, als.T)

assert np.allclose(bls[0], np.dot(X, als[0])/np.linalg.norm(np.dot(X, als[0])))

assert np.allclose(bls[0], np.dot(X, als[0])/np.sqrt(eig_valsi[0]))

#%%

# Example of chi squared test as shown on: https://online.stat.psu.edu/stat200/book/export/html/230

# this has no correlation
df_test = pd.DataFrame({'cat': [3, 4, 5],
                        'dog': [3, 4, 5],
                        'chicken': [3, 4, 5],
                        'pig': [3, 4, 5]},
                       index=['men', 'women', 'non-binary'])

# this is has high correlation
# df_test = pd.DataFrame({'cat': [0, 0, 5],
#                         'dog': [0, 5, 0],
#                         'chicken': [5, 0, 0],
#                         'pig': [1, 0, 0]},
#                        index=['men', 'women', 'non-binary'])

# this has no correlation
# df_test = pd.DataFrame({'cat': [1, 1, 1],
#                         'dog': [2, 2, 2],
#                         'chicken': [3, 3, 3],
#                         'pig': [4, 4, 4]},
#                        index=['men', 'women', 'non-binary'])

df_test['Total'] = df_test.sum(axis=1).values
df_test = df_test.append(pd.Series(df_test.sum(axis=0), name='Total'), ignore_index=False)

# E_men_cat = 12*12/48
# E_men_dog = 12*12/48
# E_men_chicken = 12*12/48
# E_men_pig = 12*12/48
#
# E_women_cat = 16*12/48
# E_women_dog = 16*12/48
# E_women_chicken = 16*12/48
# E_women_pig = 16*12/48
#
# E_nonBin_cat = 20*12/48
# E_nonBin_dog = 20*12/48
# E_nonBin_chicken = 20*12/48
# E_nonBin_pig = 20*12/48

# Same as above, but computing-smart
expected_values = dict()
total = df_test.iloc[-1:, -1:].values[0][0]
sum = 0
for ix, row in df_test.iloc[:-1:, :].iterrows():
    for col in df_test.columns[:-1:]:
        print(f'num:{df_test.loc[ix][-1:].values[0]}x{df_test[col][-1:].values[0]} denom: {total}')
        print(f'row[col]: {row[col]}')
        expect_val = (df_test.loc[ix][-1:].values[0] * df_test[col][-1:].values / total)[0]
        print(f'expect_val: {expect_val}')
        expected_values[f'E_{ix}_{col}'] = expect_val
        sum += (row[col] - expect_val)**2/expect_val

print(r'$\chi^2$: ' + f'{sum}')
