# Based on a blog post by Steve Eddins:
# http://blogs.mathworks.com/steve/2010/12/23/two-dimensional-histograms/

url = 'http://blogs.mathworks.com/images/steve/2010/mms.jpg'

import os
if not os.path.exists('mm.png'):
    print "Downloading M&M's..."
    import urllib2
    u = urllib2.urlopen(url)
    f = open('mm.png', 'w')
    f.write(u.read())
    f.close()


from skimage import io, color, exposure
import numpy as np


mm = io.imread('mm.png')
mm_lab = color.rgb2lab(mm)
L, a, b = mm_lab.T

left, right = -100, 100
bins = np.arange(left, right)
H, x_edges, y_edges = np.histogram2d(a.flatten(), b.flatten(), bins,
                                     normed=True)


import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

f = plt.figure()
ax0 = plt.subplot2grid((2, 2), (0, 0))
ax1 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))

f.suptitle('Select values by dragging the mouse on the histogram')

ax0.imshow(mm)
ax0.set_title('Input')
ax0.set_xticks([])
ax0.set_yticks([])

ax1.imshow(exposure.rescale_intensity(H, in_range=(0, 5e-4)),
           extent=[left, right, right, left], cmap=plt.cm.gray)
ax1.set_title('Histogram')
ax1.set_xlabel('b')
ax1.set_ylabel('a')

rectprops=dict(
    facecolor='gray',
    edgecolor='white',
    alpha=0.3
    )
selected_rectangle = Rectangle((0, 0), 0, 0, transform=ax1.transData,
                               **rectprops)

ax1.add_patch(selected_rectangle)

result = ax2.imshow(mm)
ax2.set_title('L + masked a, b')


def histogram_select(e_click, e_release):
    x0, y0 = e_click.xdata, e_click.ydata
    x1, y1 = e_release.xdata, e_release.ydata

    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    selected_rectangle.set_xy((x0, y0))
    selected_rectangle.set_height(y1 - y0)
    selected_rectangle.set_width(x1 - x0)

    green_mm_lab = mm_lab.copy()
    L, a, b = green_mm_lab.T

    mask = ((a > y0) & (a < y1)) & ((b > x0) & (b < x1))
    green_mm_lab[..., 1:][~mask.T] = 0

    green_mm = color.lab2rgb(green_mm_lab)

    result.set_data(green_mm)
    f.canvas.draw()

rs = RectangleSelector(ax1, histogram_select, drawtype='box',
                       spancoords='data', rectprops=rectprops)

plt.show()
