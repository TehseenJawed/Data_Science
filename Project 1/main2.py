import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# this function is the curve we wish to fit
# along with its relevant parameters
def fit(x, a, b, c, s):
    """
    a is the amplitude
    """

    return a * np.tanh(-x) * np.exp(-np.abs(x)) + c


def graph(filename, name):

    # load the data
    df = pd.read_csv(filename, header=None, skiprows=[0])
    df = df.to_numpy()

    X = df[:, 0]
    Y = df[:, 1:]

    # plot the raw data
    fig, ax = plt.subplots(figsize=(12, 8))

    for col in range(Y.shape[1]):

        # columns to skip
        if col+1 in [6, 7, 8, 9]:
            continue

        print(f"Plotting column {col+1}...", end='\r')
        ax.plot(X, Y[:, col], label=f"Plot {col+1}")

    # produce a guess for your parameters
    # NOTE manual guesses are better!
    A, B, C, S = 10, 1.2485e10, 0, 1e5

    # Gaussian best fit
    Y_hat = np.mean(Y, axis=1)

    # NOTE check out the documentation for curve_fit, its helpful
    opt, _ = curve_fit(fit, X, Y_hat, p0=[A, B, C, S])

    print("\nOptimal parameters:", opt)

    # plot the best fit
    ax.plot(X, fit(X, *opt), label=f"Best Fit", linestyle='--')

    # graph labelling
    plt.legend()
    

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(name)

    plt.savefig(f"{name}.png")
    print("Saved")


if __name__ == "__main__":
    graph("Qubit data(phase).csv", name="Phase")
