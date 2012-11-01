# Auto-clustering, suggested by Matt Terry

from skimage import io, color, exposure
from sklearn import cluster, preprocessing
import numpy as np
import matplotlib.pyplot as plt


url = 'http://blogs.mathworks.com/images/steve/2010/mms.jpg'


import os
if not os.path.exists('mm.png'):
    print "Downloading M&M's..."
    import urllib2
    u = urllib2.urlopen(url)
    f = open('mm.png', 'w')
    f.write(u.read())
    f.close()


print "Image I/O..."
mm = io.imread('mm.png')
mm_lab = color.rgb2lab(mm)
ab = mm_lab[..., 1:]

print "Mini-batch K-means..."
X = ab.reshape(-1, 2)
kmeans = cluster.MiniBatchKMeans(n_clusters=6)
y = kmeans.fit(X).labels_

labels = y.reshape(mm.shape[:2])
N = labels.max()


def no_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

# Display all clusters
for i in range(N):
    mask = (labels == i)
    mm_cluster = mm_lab.copy()
    mm_cluster[..., 1:][~mask] = 0

    ax = plt.subplot2grid((2, N), (1, i))
    ax.imshow(color.lab2rgb(mm_cluster))
    no_ticks(ax)


ax = plt.subplot2grid((2, N), (0, 0), colspan=2)
ax.imshow(mm)
no_ticks(ax)


# Display histogram

L, a, b = mm_lab.T

left, right = -100, 100
bins = np.arange(left, right)
H, x_edges, y_edges = np.histogram2d(a.flatten(), b.flatten(), bins,
                                     normed=True)

ax = plt.subplot2grid((2, N), (0, 2))
H_bright = exposure.rescale_intensity(H, in_range=(0, 5e-4))
ax.imshow(H_bright,
          extent=[left, right, right, left], cmap=plt.cm.gray)
ax.set_title('Histogram')
ax.set_xlabel('b')
ax.set_ylabel('a')


# Voronoi diagram
mid_bins = bins[:-1] + 0.5
L = len(mid_bins)

yy, xx = np.meshgrid(mid_bins, mid_bins)
Z = kmeans.predict(np.column_stack([xx.ravel(), yy.ravel()]))
Z = Z.reshape((L, L))

ax = plt.subplot2grid((2, N), (0, 3))
ax.imshow(Z, interpolation='nearest',
          extent=[left, right, right, left],
          cmap=plt.cm.Spectral, alpha=0.8)
ax.imshow(H_bright, alpha=0.2,
          extent=[left, right, right, left],
          cmap=plt.cm.gray)
ax.set_title('Clustered histogram')
no_ticks(ax)


plt.show()
