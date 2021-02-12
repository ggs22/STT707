import numpy as np
import matplotlib.pyplot as plt


X = np.random.normal(5, 5/3, (10, 3))
X[:, 0] = X[:, 0] + np.linspace(0, X.shape[0] - 1, X.shape[0])
X[:, 2] = X[:, 2] + np.linspace(0, X.shape[0] - 1, X.shape[0]) ** 2

plt.figure(figsize=(15, 15))
# plt.subplots(224)

ax1 = plt.subplot(221)
ax1.set_xlabel('pc1')
ax1.set_ylabel('pc2')
ax1.scatter(X[:, 0], X[:, 1], c='b', s=10)
for i in range(0, X.shape[0]):
    ax1.text(x=X[i, 0], y=X[i, 1], s=X[i, 0])

ax2 = plt.subplot(222)
ax2.set_xlabel('pc1')
ax2.set_ylabel('pc3')
ax2.scatter(X[:, 0], X[:, 2], c='b', s=10)
for i in range(0, X.shape[0]):
    ax2.text(x=X[i, 0], y=X[i, 2], s=X[i, 0])

ax3 = plt.subplot(223)
ax3.set_xlabel('pc2')
ax3.set_ylabel('pc3')
ax3.scatter(X[:, 1], X[:, 2], c='b', s=10)
for i in range(0, X.shape[0]):
    ax3.text(x=X[i, 1], y=X[i, 2], s=X[i, 0])

ax4 = plt.subplot(224, projection='3d')
ax4.set_xlabel('pc1')
ax4.set_ylabel('pc2')
ax4.set_zlabel('pc3')
ax4.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', s=10)
for i in range(0, X.shape[0]):
    ax4.text(x=X[i, 0], y=X[i, 1], z=X[i, 2], s=X[i, 0])

plt.show()

print(X)
