import numpy as np

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time


class Scope:
    def __init__(self, ax, N=150):
        self.ax = ax
        self.N = N

        self.xdata = np.linspace(0, 1, N)

        self.line, = self.ax.plot([], [])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1, 1)

        self.fps = 0
        self.last_render = time.time()

    def update(self, measured_data=None):
        self.line.set_data(self.xdata, measured_data)

        self.update_fps(time.time())

        return self.line,

    def measure(self):
        "Fake measuring of values"
        while True:
            yield np.sin(self.last_render * 800 * np.pi * self.xdata)

    def update_fps(self, timestamp):
        self.fps = (self.fps + 1/(timestamp - self.last_render)) / 2.
        self.last_render = timestamp


fig, ax = plt.subplots()
scope = Scope(ax)

ani = animation.FuncAnimation(fig, scope.update, scope.measure, interval=0,
                              blit=True, init_func=scope.update)
                              # when using blitting, also provide init
                              # for "blank slate"

plt.show()

print "Frames per second:", scope.fps
