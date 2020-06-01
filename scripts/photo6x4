import sys
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread(sys.argv[1])
img = np.concatenate([img] * 3, axis=1)
img = np.concatenate([img] * 2, axis=0)
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.imshow(img, aspect='auto')
fig.set_size_inches(6, 4)
plt.savefig('out.png', dpi=300)
