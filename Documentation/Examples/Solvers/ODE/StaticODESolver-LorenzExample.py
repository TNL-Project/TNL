#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np

###
# Enable latex for labels
plt.rcParams["text.usetex"] = True

###
# Parse the input file
f = open(sys.argv[1])
x_lst = []
y_lst = []
z_lst = []
for _line in f:
    _line = _line.strip()
    a = _line.split()
    x_lst.append(float(a[0]))
    y_lst.append(float(a[1]))
    z_lst.append(float(a[2]))

###
# Convert the data to NumPy array
x = np.array(x_lst)
y = np.array(y_lst)
z = np.array(z_lst)

###
# Draw the graph of u(t) using Matplotlib
fig, ax = plt.subplots()
ax.plot(x, y, label="Lorenz attractor")
ax.legend()
plt.savefig(sys.argv[2])
plt.close(fig)
