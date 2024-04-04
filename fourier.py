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
        self.coeffs = np.empty(N, dtype=complex)
        self.bases = np.empty((N, len(self.t)), dtype=complex)

        for i, k in enumerate(range(start, stop + 1)):
            base = self.bases_generator(k, self.t)
            self.bases[i] = base
            base_energy = self.bases_generator.energy(k)
            self.coeffs[i] = simpson(self.x * np.conj(base), self.t) / base_energy

    def reconstruct_signal(self):
        """
        Reconstructs the signal using the computed coefficients and bases.

        Returns:
        np.ndarray: The reconstructed signal.
        """
        return np.dot(self.coeffs, self.bases)


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
        Calculates the energy of the k-th Legendre polynomial.from fourier import FourierSeries, ExpComplex

        Parameters:
        - k (int): The order of the Legendre polynomial.

        Returns:
        float: The energy of the polynomial.
        """
        return 2 / (2 * k + 1)
    
class WalshFunctions:
    def __init__(self, T0=1):
        self.T0 = T0

    def __call__(self, k, t):
        """
        Generates the k-th Walsh function evaluated at points t using Gray codes to determine
        the sequence of Rademacher functions to multiply together. This method constructs the
        Walsh function by applying bitwise operations based on the Gray code of the order k
        and multiplying Rademacher functions accordingly.

        Parameters:
        - k (int): The order of the Walsh function, determining which Walsh function to generate.
        - t (np.ndarray): The points at which to evaluate the function, representing a continuous
          time or sample points.

        Returns:
        np.ndarray: The evaluated Walsh function at points t, representing the function's value
        at each point in t.
        """
        walsh = np.ones_like(t)
        gcode = self._get_gray_codes(k)
        for index, item in enumerate(gcode[::-1]):
            l = int(item)
            n = index + 1
            walsh *= self._rademacher(t, n)**l

        return walsh

    def _get_gray_codes(self, n):
        """
        Converts an integer to its corresponding Gray code. The Gray code is a binary numeral system
        where two successive values differ in only one bit, which is useful for minimizing errors in
        digital communication and for generating Walsh functions.

        Parameters:
        - n (int): The integer to convert to Gray code.

        Returns:
        str: The Gray code representation of the integer n as a binary string.
        """
        assert n >= 0, "Invalid input: n must be non-negative."

        n ^= (n >> 1)
        gcode = bin(n)[2:]

        return gcode
    
    def _rademacher(self, t, n):
        """
        Generates the n-th Rademacher function evaluated at points t. Rademacher functions are
        a series of orthogonal functions that take values of +1 and -1. They are used in constructing
        Walsh functions.

        Parameters:
        - t (np.ndarray): The points at which to evaluate the Rademacher function.
        - n (int): The order of the Rademacher function, determining the function's frequency.

        Returns:
        np.ndarray: The evaluated n-th Rademacher function at points t.
        """
        assert n >= 0, "Invalid input: n must be non-negative."

        if n == 0:
            rademacher = np.ones_like(t)
        else:
            rademacher = np.sign(np.sin(2 ** n * np.pi * t / self.T0))

        return rademacher

    def energy(self, k):
        """
        Calculates the energy of the k-th Walsh function. Given the orthogonal and normalized nature
        of Walsh functions, the energy of each function is considered to be 1. This is consistent with
        their application in signal processing where Walsh functions are used as a basis set for
        representing signals.

        Parameters:
        - k (int): The order of the Walsh function, although it's not directly used in this method as
          the energy of all Walsh functions is 1.

        Returns:
        float: The energy of the k-th Walsh function, which is always 1.
        """
        return self.T0
    
class ExpComplex:
    def __init__(self, T0):
        self.T0 = T0
        
    def __call__(self, k, t):
        return np.exp(2j*k*np.pi*t/self.T0)
    
    def energy(self, k):
        return self.T0
        


def square_wave(t):
    return np.sign(np.sin(t))


def triangle_wave(t):
    return signal.sawtooth(2 * t, width=0.5)


def sawtooth_wave(t):
    return signal.sawtooth(2 * t)


def main():
    # Signal parameters definition
    ti, tf, n_samples = -.5, .5, 10000
    t = np.linspace(ti, tf, n_samples)
    x = np.sin(2*np.pi*t) # signal function
    T0 = tf - ti
    # Fourier series analysis
    expcom = ExpComplex(T0)
    fs = FourierSeries(x, expcom, t)
    start, stop = -1, 1
    fs.compute_coeffs(start, stop)
    print(fs.coeffs)



if __name__ == "__main__":
    main()
