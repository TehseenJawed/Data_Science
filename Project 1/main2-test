import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt



# this function is the curve we wish to fit
# along with its relevant parameters

def gaussian(x, a, x0, sigma):
    """
    a is the amplitude
    x0 is the mean
    sigma is the standard deviation
    """
    return a * np.exp(-(x-x0)**2/(2*sigma**2))

def fit(x, a, b, c):
    """
    a is the amplitude
    """
    x = x/s - b
    return a * np.tanh(x**2) * (-1/x) + c


def graph(filename, name):

    # load the data
    df = pd.read_csv(filename, header=None, skiprows=[0])
    df = df.to_numpy()

    X = df[:, 0] / 1e9
    Y = df[:, 1:]

    # plot the raw data
    fig, ax = plt.subplots(figsize=(18, 8))

    for col in range(Y.shape[1]):

        # columns to skip
        if col+1 in [1,2,3,4,5,7, 8, 9]:
            continue

        print(f"Plotting column {col+1}...", end='\r')
        ax.scatter(X, Y[:, col], label=f"Plot {col+1}")

    # produce a guess for your parameters
    # NOTE manual guesses are better!
    A, B, C = 0, 1.236, 7.6

    # Gaussian best fit
    Y_hat = np.mean(Y, axis=1)

    # NOTE check out the documentation for curve_fit, its helpful
    opt, _ = curve_fit(gaussian, X, Y_hat, p0=[A, B, C])

    print("\nOptimal parameters:", opt)

    # plot the best fit
    ax.plot(X, gaussian(X, *opt), label=f"Best Fit", linestyle='--')

    # graph labelling
    plt.legend()
   
    plt.xlabel("Frequency (GHz)")
    plt.ylabel(name)

    plt.savefig(f"{name}.png")
    print("Done!")


if __name__ == "__main__":
    graph("Qubit data(phase).csv", name="Phase")
