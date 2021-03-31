#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.cross_decomposition import CCA

#%%
data = pd.read_excel('data_ocde.xlsx')

# exclude "TV" column
data.drop(labels=['TV'], axis=1, inplace=True)

# separate consomation and structure variables
conso_labels = ['CAL', 'LOG', 'ELEC', 'EDUC']
countries_names = data.iloc[:, 0]

X_conso_data = data[conso_labels]
Y_struct_data = data.drop(labels=conso_labels, axis=1)

# lets drop countrie's names so we have only numeric data
Y_struct_data = Y_struct_data.iloc[:, 1:]

# center enter reduce variables
X_conso_data = (X_conso_data - X_conso_data.mean(axis=0))/X_conso_data.std()
Y_struct_data = (Y_struct_data - Y_struct_data.mean(axis=0))/Y_struct_data.std()

#%%
# make canonical correlation analysis
ca = CCA()
x, y = ca.fit_transform(X_conso_data, Y_struct_data)

ca_res = pd.DataFrame({'country':countries_names,
                       'cx_0':x[:, 0],
                       'cx_1':x[:, 1],
                       'cy_0':y[:, 0],
                       'cy_1':y[:, 1]})

#%%
# exmine correlation between covariates
print(np.corrcoef(x[:, 0], y[:, 0]), '\n')
print(np.corrcoef(x[:, 1], y[:, 1]))

#%%
# plot correlations  results
fig, axes = plt.subplots(1, 2)

sns.scatterplot(x='cx_0', y='cy_0', c=[[0, 0, 1]], data=ca_res, ax=axes[0])
sns.scatterplot(x='cx_1', y='cy_1', c=[[1, 0, 0]], data=ca_res, ax=axes[1])

plt.savefig('2_corr.pdf')
plt.show()

#%%

# plot correlations results in sactter
fig = plt.figure(figsize=(8, 8))
sns.scatterplot(x='cx_0', y='cy_0', c=[[0, 0, 1]], data=ca_res)
for ix, country_name in enumerate(countries_names):
    plt.text(x=ca_res.loc[ix, 'cx_0'], y=ca_res.loc[ix, 'cy_0'] + np.power(-1, ix) * 0.05,
             s=country_name[0:3], rotation='vertical',
             size=10)
plt.savefig('corr_xy_0.pdf')
plt.show()


fig = plt.figure(figsize=(8, 8))
sns.scatterplot(x='cx_1', y='cy_1', c=[[1, 0, 0]], data=ca_res)
for ix, country_name in enumerate(countries_names):
    plt.text(x=ca_res.loc[ix, 'cx_1'], y=ca_res.loc[ix, 'cy_1'] + np.power(-1, ix) * 0.05,
             s=country_name[0:3], rotation='vertical',
             size=10)
plt.savefig('corr_xy_1.pdf')
plt.show()

 #%%

plt.figure(figsize=(8, 8))
sns.scatterplot(x='cx_0', y='cx_1', c=[[0, 0, 1]], data=ca_res)
for ix, country_name in enumerate(countries_names):
    plt.text(x=ca_res.loc[ix, 'cx_0'], y=ca_res.loc[ix, 'cx_1'], s=country_name, size=10)
plt.show()

#%%

plt.figure(figsize=(8, 8))
sns.scatterplot(x='cy_0', y='cy_1', c=[[1, 0, 0]], data=ca_res)
for ix, country_name in enumerate(countries_names):
   plt.text(x=ca_res.loc[ix, 'cy_0'], y=ca_res.loc[ix, 'cy_1'], s=country_name, size=10)
plt.show()

#%%

# plot correlations heatmaps
corr_x_df = pd.concat((ca_res[['cx_1', 'cx_0']], X_conso_data), axis=1)
corr_y_df = pd.concat((ca_res[['cy_1', 'cy_0']], Y_struct_data), axis=1)

sns.heatmap(corr_x_df.corr().where(np.tril(np.ones(corr_x_df.corr().shape)).astype(bool)),
            cmap='RdYlBu', vmin=-1, vmax=1)
plt.savefig('corr_hm_y.pdf')
plt.show()

sns.heatmap(corr_y_df.corr().where(np.tril(np.ones(corr_y_df.corr().shape)).astype(bool)),
            cmap='RdYlBu', vmin=-1, vmax=1)
plt.savefig('corr_hm_yX.pdf')
plt.show()

#%%

print(countries_names[Y_struct_data.sort_values(by='FBCF', ascending=False).index])
countries_names[Y_struct_data.sort_values(by='RECC', ascending=True).index]

# create a composite latent variable from the correlation seen in the heat maps
Y_struct_data['lat_var'] = Y_struct_data['PNB'] + Y_struct_data['RECC'] - Y_struct_data['FBCF']
X_conso_data['lat_var'] = X_conso_data['CAL'] + X_conso_data['EDUC'] - X_conso_data['LOG']

# plot correlations  results

fig = plt.figure(figsize=(8, 8))
for ix, name_ix in enumerate(Y_struct_data.sort_values(by='RECC', ascending=False).index):
    plt.scatter(x=ca_res.loc[name_ix, 'cx_0'], y=ca_res.loc[name_ix, 'cy_0'], c=[[1, (1/18) * ix, 0]])
    plt.text(x=ca_res.loc[name_ix, 'cx_0'], y=ca_res.loc[name_ix, 'cy_0'] + np.power(-1, ix) * 0.05,
             s=countries_names[name_ix][0:3], rotation='vertical',
             size=10)
plt.savefig('scattered_cons_correlation.pdf')
plt.show()

#%%
fig = plt.figure(figsize=(8, 8))
for ix, name_ix in enumerate(Y_struct_data.sort_values(by='lat_var', ascending=False).index):
    plt.scatter(x=ca_res.loc[name_ix, 'cx_0'], y=ca_res.loc[name_ix, 'cy_0'], c=[[1, (1/18) * ix, 0]])
    plt.text(x=ca_res.loc[name_ix, 'cx_0'], y=ca_res.loc[name_ix, 'cy_0'] + np.power(-1, ix) * 0.05,
             s=countries_names[name_ix][0:3], rotation='vertical',
             size=10)
plt.savefig('scattered_struc_correlation.pdf')
plt.show()

#%%
plt.figure(figsize=(8, 8))

print(countries_names[Y_struct_data.sort_values(by='lat_var', ascending=True).index])

for ix, ix_name in enumerate(Y_struct_data.sort_values(by='lat_var', ascending=True).index):
    plt.scatter(x=ca_res.loc[ix_name, 'cx_0'], y=ca_res.loc[ix_name, 'cx_1'], c=[[1, (1/18) * ix, 0]])
    plt.text(x=ca_res.loc[ix_name, 'cx_0'], y=ca_res.loc[ix_name, 'cx_1'],
             s=countries_names[ix_name][0:3], size=10)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.savefig('scatter_struc_lat_var_coloration.pdf')
plt.show()

#%%
plt.figure(figsize=(8, 8))

print(countries_names[X_conso_data.sort_values(by='lat_var', ascending=True).index])

for ix, ix_name in enumerate(X_conso_data.sort_values(by='lat_var', ascending=True).index):
    plt.scatter(x=ca_res.loc[ix_name, 'cx_0'], y=ca_res.loc[ix_name, 'cx_1'], c=[[1, (1/18) * ix, 0]])
    plt.text(x=ca_res.loc[ix_name, 'cx_0'], y=ca_res.loc[ix_name, 'cx_1'],
             s=countries_names[ix_name][0:3], size=10)

plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.savefig('scatter_cons_lat_var_coloration.pdf')
plt.show()
