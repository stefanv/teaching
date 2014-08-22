import numpy as np

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time


class Scope:
    def __init__(self, ax, N=150):
        self.ax = ax
        self.N = N

        self.data = np.zeros((255, 255), dtype=np.uint8)

        self.image = plt.imshow(self.data, vmin=0, vmax=255, cmap='gray')

        self.fps = 0
        self.last_render = time.time()

    def update(self, measured_data=None):
        self.data += 1
        self.image.set_data(self.data)

        self.update_fps(time.time())

        return self.image,

    def update_fps(self, timestamp):
        self.fps = (self.fps + 1/(timestamp - self.last_render)) / 2.
        self.last_render = timestamp

    def measure(self):
        "Fake measuring of values"
        while 1:
            yield 1


fig, ax = plt.subplots()
scope = Scope(ax)

ani = animation.FuncAnimation(fig, scope.update, scope.measure, interval=10,
                              blit=True, init_func=scope.update)
                              # when using blitting, also provide init
                              # for "blank slate"

plt.show()

print "Frames per second:", scope.fps
