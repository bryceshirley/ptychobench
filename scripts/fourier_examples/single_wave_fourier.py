# gratings.py

import numpy as np
import matplotlib.pyplot as plt


def grating_function(wavelength, angle):
    x = np.arange(-500, 501, 1)

    X, Y = np.meshgrid(x, x)
    grating = np.sin(2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength)

    plt.set_cmap("gray")

    plt.subplot(121)
    plt.imshow(grating)
    plt.title(f"Original Sine Function\n frequency = {1 / wavelength}, Angle = {angle}")
    plt.xticks([])
    plt.yticks([])

    # Calculate Fourier transform of grating
    ft = np.fft.ifftshift(grating)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)

    plt.subplot(122)
    plt.imshow(abs(ft))
    plt.title("Fourier Transform")
    plt.xlim([480, 520])
    plt.ylim([520, 480])  # Note, order is reversed for y
    # Add centre cross
    plt.axhline(y=500, color="r", linestyle="--")
    plt.axvline(x=500, color="r", linestyle="--")
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

    plt.show()


if __name__ == "__main__":
    grating_function(200, 0)
    grating_function(100, 0)
    grating_function(200, np.pi / 4)
