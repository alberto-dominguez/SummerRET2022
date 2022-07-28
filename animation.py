import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

x = range(100)
y = np.random.rand(100)

fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
graph, = plt.plot([], [], '-')


def init():
    return graph,


def animate(i):
    graph.set_data(x[:i], y[:i])
    return graph,


ani = FuncAnimation(fig, animate, frames=range(len(x)), interval=50, save_count=len(x), init_func=init, blit=True)
ani.save('ani.gif', writer=PillowWriter(fps=30))
