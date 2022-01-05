#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize as opt

# User defined inputs required for analysis, check these are correct
# for your experiment!
slit_distance = 75.0e-3  # Distance from slit to CCD, you have to edit this [m]
wavelength = 0.6328e-6  # Wavelength of laser [m]
pixel_size = 1.e-6  # Pixel size, you have to edit this [m]

estimated_slit_width = 100.e-6  # Give rough estimate of slit width [m]
estimated_intensity = 1.  	# Should be ~1 from normalisation (see later)

# Read in image file as a 2D array.
# You will need to change this to read in your file
img = mpimg.imread('./FD_test_25.05.2021_100um.tif')


# Sum all rows in 2D array to make 1D intensity plot as a function of pixel
# number
recorded_intensity = np.sum(img, axis=0).tolist()

# Plot intensity as a function of theta
n_pix = len(recorded_intensity)
theta = []
max_intensity = np.max(recorded_intensity)
normalised_intensity = recorded_intensity/max_intensity  # Normalise spectrum

# Calculate theta
# You will need to edit this to get values of theta
for i in range(0, n_pix):
    exit('You have to calculate theta here.')  # Remove this
    theta.append()  # Append your theta calculation to the list


# Define fit function
# Note that numpy's sinc function already has a factor of pi, such that
# np.sinc(x) is equal to sin(pi*x)/(pi*x)
# https://numpy.org/doc/stable/reference/generated/numpy.sinc.html
# You will need to edit this to include the wavelength and slit width as
# arguments and use them in the function. You might also want to consider a
# contribution from background.
def sinc2_func(x, intensity, constant):
    psi = constant * np.sin(np.array(x))
    return intensity * (np.sinc(psi))**2


# Fit diffraction pattern from normalised_intensity as a function of theta with
# your function. This returns a list "params" containing the results from the
# fit, and a matrix "params_covariance" which contains the covariance matrix
# (uncertainties) from the fit. The uncertainties on each parameter is equal
# to the square root of the diagonal components (i.e. uncertainty on parameter
# 1 is equal to square root of element [1,1] in matrix).
#
# Here, the list p0 initialises the parameters of the fit with initial guesses
# as their values, so "intensity" is initialised as =1., and "constant"=63.
# These initial "guesses" help the fit function to minimise. If you change
# the number of parameters in you fit function you will need to edit the p0
# list initial values to reflect this, and use your parameters for the
# "estimated_slit_width", etc.
params, params_covariance = opt.curve_fit(sinc2_func,
                                          theta,
                                          normalised_intensity,
                                          p0=[estimated_intensity,
                                              estimated_slit_width])


# Output results of fit to terminal
print('Results from fit:', params)
print('Covariance matrix from fit:', params_covariance)

# Draw a plot of "normalised_intensity" as a function of "theta", with
# the fit on top.
plt.figure(figsize=(6, 4))
plt.scatter(theta, normalised_intensity, label='Data', s=0.5)
plt.plot(theta, sinc2_func(theta, *params), label='Fitted function', c='r')
plt.legend(loc='best')
plt.show()
