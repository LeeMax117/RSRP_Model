import random
import matplotlib.pyplot as plt
import os
import numpy as np

Need_Compute_Ignore_Path = False


def load_or_compute_ignore_path(file):
    if os.path.exists('ignore.csv'):
        npdata = np.loadtxt('ignore.csv', skiprows=0, dtype=int, usecols=(0, 1), delimiter=',')
        ignore_set = set(zip(npdata[:, 0], npdata[:, 1]))
        return ignore_set
    elif os.path.exists(file):
        npdata = np.loadtxt(file, skiprows=1, dtype=int, usecols=(0, 1), delimiter=',')
        max_x, max_y = np.max(npdata[:, 0]), np.max(npdata[:, 1])
        allset = set([(i, j) for i in range(max_x) for j in range(max_y)])
        ignore_set = allset.difference(set(zip(npdata[:, 0], npdata[:, 1])))
        r, c = zip(*ignore_set)
        new_data = np.c_[np.array(r), np.array(c)]
        np.savetxt('ignore.csv', new_data, fmt='%d', delimiter=',')
        return ignore_set


def random_walk(n):
    global x, y
    for i in range(n):
        x_old, y_old = x, y
        [direction] = random.choices(['N', 'S', 'E', 'W'], weights=energy)
        if direction == 'E':
            if x + 1 > 300:
                x = 300
            else:
                x += 1
                energy[2] += delta
        elif direction == 'W':
            if x - 1 < 0:
                x = 0
            else:
                x -= 1
                energy[3] += delta
        elif direction == 'N':
            if y + 1 >150:
                y = 150
            else:
                y += 1
                energy[0] += delta
        elif direction == 'S':
            if y - 1 < 0:
                y = 0
            else:
                y -= 1
                energy[1] += delta
        if (x, y) in ignore_path:
            x, y = x_old, y_old
        else:
            ignore_path.add((x, y))
            path_list.append((x, y))


energy = [1, 1, 1, 1] # ['N', 'S', 'E', 'W']
train_filename = 'train.csv'
if Need_Compute_Ignore_Path:
    ignore_path = load_or_compute_ignore_path(train_filename)
else:
    ignore_path = set()
x = random.randint(0, 300)
y = random.randint(0, 150)
path_list = list()
delta = 3
# walk the blocks
random_walk(206)
if len(path_list) > 0:
    X, Y = zip(*path_list)
    plt.figure()
    plt.xlim(-5, 320)
    plt.ylim(-5, 167)
    plt.scatter(X, Y)
    plt.show()
else:
    print('no walks')
