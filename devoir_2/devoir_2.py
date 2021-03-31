#%%
import numpy as np
import matplotlib.pyplot as plt


def plot_initial_context(plot_xs=False):
    plt.scatter(g1[0], g1[1], marker='x', s=100, color='r')
    plt.scatter(g2[0], g2[1], marker='x', s=100, color='r')
    plt.text(g1[0], g1[1] + 0.2, 'g1')
    plt.text(g2[0], g2[1] + 0.2, 'g2')
    if plot_xs:
        plt.scatter(x1[0], x1[1], marker='.', color='g')
        plt.scatter(x2[0], x2[1], marker='.', color='g')
        plt.text(x1[0], x1[1], s='x1')
        plt.text(x2[0], x2[1], s='x2')
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')


def get_frontier_1b():
    return np.sqrt(np.power(x, 2)/2 + 18) - 5


# def get_frontier_1b_negative():
#     return -np.sqrt(0.5 * np.power(x, 2) + 18) - 5


x = np.linspace(-10, 10, 41)

g1 = np.array([1, 0])
g2 = np.array([-2, 0])

M1 = np.array([[1, 0], [0, 1]])
M2 = np.array([[2, 0], [0, 1/2]])

x1 = np.array([0, 4])
x2 = np.array([-1, 3])

# Q 1 a)
plot_initial_context()
plt.axvline(x=-0.5, color='b', linestyle=':')
plt.text(x=-.4, y=1, s=r'$x_1$= -1/2', c='b')
plt.text(x=-4.5, y=4, s=r'$\Gamma_2 =\{(x_1, x_2):x_1 < -\frac{1}{2}\}$', c='b')
plt.text(x=0.5, y=-4, s=r'$\Gamma_2 =\{(x_1, x_2):x_1 > -\frac{1}{2}\}$', c='b')
plt.savefig('q1_a.pdf')
plt.show()

# Q 1 b)
plot_initial_context()
f3 = get_frontier_1b()
plt.plot(f3, x, c='b', linestyle=':')
plt.text(x=-.6, y=1, s=r'$x_1 = \sqrt{\frac{1}{2}(x_2)^2+18}-5$', c='b')
plt.text(x=-4.5, y=4, s=r'$\Gamma_2 =\{(x_1, x_2):x_1 < \sqrt{\frac{1}{2}(x_2)^2+18}-5\}$', c='b')
plt.text(x=0.5, y=-4, s=r'$\Gamma_2 =\{(x_1, x_2):x_1 > \sqrt{\frac{1}{2}(x_2)^2+18}-5\}$', c='b')
plt.savefig('q1_b.pdf')
plt.show()

# Q 2

# Q 1 a)
plot_initial_context(plot_xs=True)
plt.axvline(x=-0.5, color='b', linestyle=':')
plt.savefig('q2_a.pdf')
plt.show()

# Q 1 b)
plot_initial_context(plot_xs=True)
f3 = get_frontier_1b()
plt.plot(f3, x, c='b', linestyle=':')
plt.savefig('q2_b.pdf')
plt.show()


#%%

poids = np.array([1, 2, 3])
tailles = np.array([2, 4, 6])

plt.scatter(poids, tailles)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(xs=poids[0], ys=poids[1], zs=poids[2])
ax.scatter(xs=tailles[0], ys=tailles[1], zs=tailles[2])
plt.show()

#%%


mu1 = np.array([0, 2])
mu2 = np.array([-2, 0])
mu3 = np.array([2, 0])

sigma = 1

x1 = np.linspace(-10, 0.22907, 41)
x2 = np.linspace(0.22907, 10, 41)
y1 = 2 * x2 + sigma * (np.log(.2/.3))
y2 = -2 * x1 + sigma * (np.log(.5/.3))


plt.scatter(mu1[0], mu1[1], c='b')
plt.scatter(mu2[0], mu2[1], c='r')
plt.scatter(mu3[0], mu3[1], c='g')

plt.text(x=mu1[0], y=mu1[1]+.1, s=r'$\mu_1$')
plt.text(x=mu1[0]-.6, y=mu1[1]+.4, s=r'$R_1$', size=16)
plt.text(x=mu3[0], y=mu3[1]+.1, s=r'$\mu_2$')
plt.text(x=mu2[0], y=mu2[1]+.1, s=r'$\mu_3$')

plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

plt.plot(x2, y1, c='b', linestyle='--')
plt.plot(x1, y2, c='b', linestyle='--')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.savefig('Q2_1.pdf')
plt.show()
