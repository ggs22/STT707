import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def formater(val: float):
    return f'{val:.2f}'


X = pd.read_excel('data_ocde.xlsx')

print(X.describe().to_string(formatters=[formater] * (X.shape[1] - 1)))

X_struct = X.iloc[:, 0:14]
X_consom = X.iloc[:, 14:19]

scaler = StandardScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X.iloc[:, 1:]), columns=X.columns[1:])
X_scaled_struct = X_scaled.iloc[:, 0:13].copy()
X_scaled_consom = X_scaled.iloc[:, 13:17].copy()

corr = np.corrcoef(X_scaled)
corr_struct = np.corrcoef(X_scaled_struct)
corr_consom = np.corrcoef(X_scaled_consom)

pca = PCA(n_components=X_scaled.shape[1])
pca_struct = PCA(n_components=X_scaled_struct.shape[1])
pca_consom = PCA(n_components=X_scaled_consom.shape[1])

X_pca = pca.fit_transform(X_scaled)
X_pca_struct = pca_struct.fit_transform(X_scaled_struct)
X_pca_consom = pca_consom.fit_transform(X_scaled_consom)


def plot_variance_ratios(_pca: sk.decomposition.PCA, title=''):
    x = np.linspace(1, len(_pca.explained_variance_ratio_), len(_pca.explained_variance_ratio_))
    plt.bar(x, _pca.explained_variance_ratio_)
    plt.plot(x, np.divide(np.max(_pca.explained_variance_ratio_), x))
    plt.title(title)
    plt.xlabel('composante')
    plt.ylabel('variance expliquée')
    _suffix = title[title.find('\n') + 1:]
    plt.savefig(f'Variance_ratios_{_suffix}.pdf')
    plt.show()


plot_variance_ratios(pca,
                     title='Ratio de la variance expliquée par composante principale\n(Ensemble de données)')
plot_variance_ratios(pca_struct,
                     title='Ratio de la variance expliquée par composante principale\n(Variables de structure)')
plot_variance_ratios(pca_consom,
                     title='Ratio de la variance expliquée par composante principale\n(Variable de consomation)')


def plot_projections(data_set: np.ndarray, title='', c_means=None, labels=None):
    # plt.figure(figsize=(15, 15))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title, fontsize=12)

    projections_2d = (axes[0, 0], axes[0, 1], axes[1, 0])
    c_indexes = np.array([[1, 2], [3, 2], [1, 3]])

    for ax, ix in zip(projections_2d, c_indexes):
        ax.set_xlabel(f'pc{ix[0]}')
        ax.set_ylabel(f'pc{ix[1]}')
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.scatter(data_set[:, ix[0] - 1], data_set[:, ix[1] - 1], s=25, c=labels, cmap='tab10')

        for i in range(0, X.shape[0]):
            ax.text(x=data_set[i, ix[0] - 1], y=data_set[i, ix[1] - 1], s=f'{i}', fontsize=10)

        if c_means is not None:
            ax.scatter(c_means[:, ix[0] - 1], c_means[:, ix[1] - 1], s=500, marker='+', c=range(0, c_means.shape[0]),
                       cmap='tab10')
            for inc, cls in enumerate(range(0, c_means.shape[0])):
                ax.text(x=c_means[cls, ix[0] - 1], y=c_means[cls, ix[1] - 1], s=f'C{inc}', fontsize=10)

    ax4 = plt.subplot(224, projection='3d')
    ax4.set_xlabel('pc1')
    ax4.set_ylabel('pc2')
    ax4.set_zlabel('pc3')
    ax4.scatter(data_set[:, 0], data_set[:, 1], data_set[:, 2], s=25, c=labels, cmap='tab10')

    for i in range(0, X.shape[0]):
        ax4.text(x=data_set[i, 0], y=data_set[i, 1], z=data_set[i, 2], s=f'{i}', fontsize=8)

    _suffix = title[title.find('\n') + 1:]
    plt.savefig(f'pc_projections_{_suffix}.pdf')
    plt.show()



prefix = 'Projection 2D des données sur les composantes principales'
for ds, t in zip([X_pca, X_pca_struct, X_pca_consom],
                 ['Ensemble des variables', 'Variables de structure', 'Variables de consomation']):
    plot_projections(ds, title=f'{prefix}\n({t})')
del prefix

print(X)

# %%
# Classer les pays selon leurs consomations

consom_index = X_consom.sum(axis=1)
consom_stats = consom_index.describe()

X_scaled_struct.loc[:, 'consomation_class'] = 0

X_scaled_struct.loc[consom_index <= consom_stats['25%'],
                    'consomation_class'] = 0
X_scaled_struct.loc[(consom_index > consom_stats['25%']) & (consom_index <= consom_stats['50%']),
                    'consomation_class'] = 1
X_scaled_struct.loc[(consom_index > consom_stats['50%']) & (consom_index <= consom_stats['75%']),
                    'consomation_class'] = 2
X_scaled_struct.loc[consom_index > consom_stats['75%'],
                    'consomation_class'] = 3

classes = np.sort(X_scaled_struct['consomation_class'].unique())

classes_mean = pd.DataFrame()
for classe in classes:
    classes_mean = classes_mean.append((X_scaled_struct[X_scaled_struct['consomation_class'] == classe].mean(axis=0)),
                                       ignore_index=True)

classes_means_pca = pca_struct.transform(classes_mean.drop(['consomation_class'], axis=1))

plot_projections(X_pca_struct, title='Variables de structure avec variables de \n consomation en supplémentaie',
                 c_means=classes_means_pca,
                 labels=X_scaled_struct['consomation_class'])
