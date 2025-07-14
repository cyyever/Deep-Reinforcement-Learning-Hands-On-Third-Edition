#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

MEAN = 10
SAMPLES = 1000000
X_MAX = MEAN * 2
BINS = 100
STEP = X_MAX / BINS


if __name__ == "__main__":
    c1, c2, c5 = [0] * BINS, [0] * BINS, [0] * BINS
    for _ in range(SAMPLES):
        v1 = np.random.normal(loc=MEAN, scale=1.0)
        v2 = np.random.normal(loc=MEAN, scale=2.0)
        v5 = np.random.normal(loc=MEAN, scale=5.0)
        if 0 <= v1 <= X_MAX:
            b = int(BINS * v1 / X_MAX)
            c1[b] += 1
        if 0 <= v2 <= X_MAX:
            b = int(BINS * v2 / X_MAX)
            c2[b] += 1
        if 0 <= v5 <= X_MAX:
            b = int(BINS * v5 / X_MAX)
            c5[b] += 1
    x = [STEP * i for i in range(BINS)]
    y1 = [c / SAMPLES for c in c1]
    y2 = [c / SAMPLES for c in c2]
    y5 = [c / SAMPLES for c in c5]
    print(x)
    print(y1)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, color='black', linewidth=1.2, linestyle='-')
    ax1.plot(x, y2, color='black', linewidth=1.2, linestyle=':')
    ax1.plot(x, y5, color='black', linewidth=1.2, linestyle='--')
    ax1.legend(["Variance = 1.0", "Variance = 2.0", "Variance = 5.0"],
               loc='upper right', fancybox=True)
    ax1.set_xlim(0, X_MAX)
    ax1.grid(True, axis='both')
    ax1.set_title("Gaussian Distribution with mean = 10.0")
    plt.savefig("norm_dist.svg")
    pass