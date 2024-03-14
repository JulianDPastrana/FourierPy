import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import eval_legendre
from scipy.integrate import simpson


class FourierSeries:
    """
    Performs Fourier series analysis on a given signal by calculating
    coefficients for a specified set of basis functions and reconstructing
    the signal using these bases.
    """

    def __init__(self, x, bases_generator, t):
        """
        Initializes the Fourier Series.

        Parameters:
        - x (np.ndarray): The input signal.
        - bases_generator (callable): Function generator for orthogonal bases.
        - t (np.ndarray): Array of time values corresponding to the signal.
        """
        self.x = x
        self.t = t
        self.bases_generator = bases_generator
        self.coeffs = None
        self.bases = None

    def compute_coeffs(self, start, stop):
        """
        Computes Fourier coefficients for the first N bases.

        Parameters:
        - start (int): Start index of basis functions.
        - stop (int): Stop index of basis functions.
        """
        self.start = start
        self.stop = stop
        N = stop - start + 1
        self.coeffs = np.empty(N)
        self.bases = np.empty((N, len(self.t)))

        for i, k in enumerate(range(start, stop + 1)):
            base = self.bases_generator(k, self.t)
            self.bases[i] = base
            base_energy = self.bases_generator.energy(k)
            self.coeffs[i] = simpson(self.x * base, self.t) / base_energy

    def reconstruct_signal(self):
        """
        Reconstructs the signal using the computed coefficients and bases.

        Returns:
        np.ndarray: The reconstructed signal.
        """
        return np.dot(self.coeffs, self.bases)

    def plot(self):
        """
        Plots the original signal, reconstructed signal, and Fourier coefficients.
        """
        plt.figure(figsize=(14, 7))

        # Original and reconstructed signal
        plt.subplot(1, 2, 1)
        plt.plot(self.t, self.x, label="Original Signal")
        plt.plot(
            self.t,
            self.reconstruct_signal(),
            label="Reconstructed Signal",
            linestyle="--",
        )
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title("Fourier Series Reconstruction")

        # Fourier coefficients
        plt.subplot(1, 2, 2)
        plt.stem(range(self.start, self.stop + 1), self.coeffs)
        plt.xlabel("Basis Function Index")
        plt.ylabel("Coefficient Magnitude")
        plt.title("Fourier Coefficients")

        plt.tight_layout()
        plt.savefig("./figure.png")
        plt.close()


class LegendrePoly:
    """
    Represents Legendre polynomials used as basis functions for Fourier series.
    """

    def __call__(self, k, t):
        """
        Evaluates the k-th Legendre polynomial at points t.

        Parameters:
        - k (int): The order of the Legendre polynomial.
        - t (np.ndarray): The points at which to evaluate the polynomial.

        Returns:
        np.ndarray: Evaluated polynomial at points t.
        """
        return eval_legendre(k, t)

    def energy(self, k):
        """
        Calculates the energy of the k-th Legendre polynomial.

        Parameters:
        - k (int): The order of the Legendre polynomial.

        Returns:
        float: The energy of the polynomial.
        """
        return 2 / (2 * k + 1)


def square_wave(t):
    return np.sign(np.sin(t))


def triangle_wave(t):
    return signal.sawtooth(2 * t, width=0.5)


def sawtooth_wave(t):
    return signal.sawtooth(2 * t)


def main():
    # Signal parameters definition
    ti, tf, n_samples = -1, 1, 1000
    t = np.linspace(ti, tf, n_samples)
    x = square_wave(t) # signal function
    Ex = 2 # signal energy

    # Fourier series analysis
    poly = LegendrePoly()
    fs = FourierSeries(x, poly, t)
    start, stop = 0, 13
    fs.compute_coeffs(start, stop)
    fs.plot()

    # Cumulative energy calculation
    Ep = sum(
        fs.coeffs[i] ** 2 * poly.energy(k) for i, k in enumerate(range(start, stop + 1))
    )
    print(f"Cumulative Energy: {Ep:.3f} - {Ep/Ex:.2%}")


if __name__ == "__main__":
    main()
