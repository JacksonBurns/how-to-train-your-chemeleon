import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

df = pd.read_csv('autoresearch.csv')
n = len(df)
x = np.arange(1, n + 1)
y = df['val_mse'].values

fig, ax = plt.subplots(figsize=(10, 5))

cummin = np.minimum.accumulate(y)
rolled = np.roll(cummin, 1)
rolled[0] = cummin[0]
pareto_mask = (y == cummin) & ((x == 1) | (y < rolled))
pareto_x = x[pareto_mask]
pareto_y = y[pareto_mask]
discarded_x = x[~pareto_mask]
discarded_y = y[~pareto_mask]

ax.plot(pareto_x, pareto_y, color='green', linewidth=2, zorder=2, label='pareto')
ax.scatter(discarded_x, discarded_y, c='lightgreen', s=30, zorder=3, edgecolors='white', linewidths=0.5, label='discarded')
ax.scatter(pareto_x, pareto_y, c='green', s=50, zorder=4, edgecolors='white', linewidths=1, label='kept')

ax.set_yscale('linear')
ax.set_xlabel('Run', fontsize=12)
ax.set_ylabel('Val MSE', fontsize=12)
ax.set_xticks(x)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
ax.grid(True, axis='x', linestyle='--', alpha=0.15, color='gray')
ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('progress.png', dpi=150, bbox_inches='tight')
print('Saved progress.png')
