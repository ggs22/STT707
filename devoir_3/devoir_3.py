# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

mu1 = 0
sig1 = 1

mu2 = 4
sig2 = 3

x = np.linspace(-7, 10, 810)
ticks = np.linspace(-7, 10, 18)

y1 = norm.pdf(x=x, loc=mu1, scale=sig1)
y2 = norm.pdf(x=x, loc=mu2, scale=sig2)

plt.figure(figsize=(6.5, 6.5))

plt.plot(x, y1, c='r', label='fct. densité de \n probabilité de $C_1$')
plt.plot(x, y2, c='b', label='fct. densité de \n probabilité de $C_2$')

plt.axvline(x=-2.6729, c='g', linestyle=':')
plt.axvline(x=1.6730, c='g', linestyle=':')

plt.axhline(y=0, c='k', linewidth=0.5)

plt.vlines(x=mu1, ymin=0, ymax=max(y1), colors='r', linestyle='--')
plt.vlines(x=mu2, ymin=0, ymax=max(y2), colors='b', linestyle='--')

plt.text(x=-6, y=0.3, s=r'x$\approx$-2.67')
plt.text(x=1.6730, y=0.3, s=r'x$\approx$1.67')

plt.xlabel('x')
plt.ylabel('P(x)')
plt.xticks(ticks=ticks)

plt.legend(loc='upper left')
plt.savefig('Q3.pdf')
plt.show()
