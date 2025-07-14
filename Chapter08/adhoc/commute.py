import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.clf()
    v1 = np.random.normal(30, 2.0, size=2000)
    v2 = np.random.normal(90, 4.0, size=200)
    v = np.concatenate((v1, v2))
    mean_time = v.mean()
    plt.hist(v, normed=True, bins=100)
    plt.title(f"Car commute time distribution\nmean={mean_time:.2f} mins")
    plt.xlabel("Time, minutes")
    plt.ylabel("Probability")
    plt.savefig("commute-car.png")

    plt.clf()
    v1 = np.random.normal(40, 2.0, size=2000)
    v2 = np.random.normal(60, 1.0, size=50)
    v = np.concatenate((v1, v2))
    mean_time = v.mean()
    plt.hist(v, normed=True, bins=100)
    plt.title(f"Train commute time distribution\nmean={mean_time:.2f} mins")
    plt.xlabel("Time, minutes")
    plt.ylabel("Probability")
    plt.savefig("commute-train.png")
