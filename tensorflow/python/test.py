from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import random
import time


fig = plt.figure()     #figure(도표) 생성

ax = plt.subplot(211, xlim=(0, 50), ylim=(0, 1024))
ax_2 = plt.subplot(212, xlim=(0, 50), ylim=(0, 512))


max_points = 50
max_points_2 = 50

line, = ax.plot(np.arange(max_points),
                np.ones(max_points, dtype=np.float)*np.nan, lw=1, c='blue',ms=1)
line_2, = ax_2.plot(np.arange(max_points_2),
                np.ones(max_points, dtype=np.float)*np.nan, lw=1,ms=1)


def init():
    return line
def init_2():
    return line_2


def animate(i):
    y = random.randint(0,1024)
    old_y = line.get_ydata()
    new_y = np.r_[old_y[1:], y]
    line.set_ydata(new_y)
    print(new_y)
    return line
#anim_2 = animation.FuncAnimation(fig, animate_2  , init_func= init_2 ,frames=200, interval=10, blit=False)
def animate_2(i):
    y_2 = random.randint(0,512)
    old_y_2 = line_2.get_ydata()
    new_y_2 = np.r_[old_y_2[1:], y_2]
    line_2.set_ydata(new_y_2)
    print(new_y_2)
    return line_2


anim = animation.FuncAnimation(fig, animate  , init_func= init ,frames=400, interval=50, blit=False)
anim_2 = animation.FuncAnimation(fig, animate_2  , init_func= init_2 ,frames=200, interval=10, blit=False)
plt.show()
