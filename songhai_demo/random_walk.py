import random
import matplotlib.pyplot as plt


energy = [1, 1, 1, 1] # ['N', 'S', 'E', 'W']
x = random.randint(0, 300)
y = random.randint(0, 150)
ignore_path = set()
path_list = list()
delta = 3

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


random_walk(206)
X, Y = zip(*path_list)
plt.figure()
plt.xlim(0, 300)
plt.ylim(0, 150)
plt.scatter(X, Y)
plt.show()