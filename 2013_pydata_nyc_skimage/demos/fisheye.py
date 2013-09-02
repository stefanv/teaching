from skimage import transform, data

import numpy as np
import matplotlib.pyplot as plt

image = data.lena()

def fisheye(xy):
    center = np.mean(xy, axis=0)
    xc, yc = (xy - center).T

    # Polar coordinates
    r = np.sqrt(xc**2 + yc**2)
    theta = np.arctan2(yc, xc)

    r = 0.87 * np.exp(r**(1/2.5) / 1.75)

    return np.column_stack((
        r * np.cos(theta), r * np.sin(theta)
        )) + center

out = transform.warp(image, fisheye)

f, (ax0, ax1) = plt.subplots(1, 2,
                             subplot_kw=dict(xticks=[], yticks=[]))
ax0.imshow(image)
ax1.imshow(out)

plt.show()
