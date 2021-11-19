import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize as opt

# User defined inputs required for analysis, check these are correct for your experiment!
slitDistance = 75.0		# Distance from slit to CCD, you have to edit this [mm]
waveLength   = 0.6328		# Wavelength of laser [um]
pixelSize    = 1.			# Pixel size, you have to edit this [um]

estimatedSlitWidth = 100.	# Give rough estimate of slit width [um]
estimatedIntensity = 1.  	# Should be ~1 from normalisation (see later)

# Read in image file as a 2D array.
# You will need to change this to read in your file
img = mpimg.imread('./FD_test_25.05.2021_100um.tif')
#print(img)         # print array to terminal
#imgplot = plt.imshow(img)   # Plot 2D image as colour map

# Sum all rows in 2D array to make 1D intensity plot as a function of pixel number
recordedIntesity = np.sum(img,axis=0).tolist()

# Plot intensity as a function of theta
nPix = len(recordedIntesity)
theta = []
maxIntensity = np.max(recordedIntesity)
normalisedIntensity = [] 

# Calculate theta and normalise spectrum so max value = 1
# You will need to edit this to get values of theta
for i in range(0,nPix):
	theta.append( i )
	normalisedIntensity.append(float(recordedIntesity[i])/maxIntensity)


# Define fit function
# Note that python's sinc function already has a factor of pi, such that
# it np.sinc(x) is equal to sin(pi*x)/(pi*x) https://numpy.org/doc/stable/reference/generated/numpy.sinc.html
# You will need to edit this to include the wavelength and slit width as arguments and use them in the 
# function. You might also want to consider a contribution from background.
def sinc2_func(x, intensity, constant):
	psi = constant * np.sin(np.array(x))
	return intensity * (np.sinc(psi))**2


# Fit diffraction pattern from normalisedIntensity as a function of theta with your function.
# This returns a list "params" containing the results from the fit, and a matrix "params_covariance"
# which contains th covariance matrix (uncertainties) from the fit. The uncertainties on each parameter 
# is equal to the square root of the diagonal components (i.e. uncertainty on parameter 1 is equal to square
# root of element [1,1] in matrix).
#
# Here, the list p0 initialises the parameters of the fit with initial guesses as their values, so 
# "intensity" is initialised as =1., and "constant"=63. These initial "guesses" help the fit function to 
# minimise. If you change the number of parameters in you fit function you will need to edit the p0 list 
# initial values to reflect this, and use your parameters for the "estimatedSlitWidth", etc.
params, params_covariance = opt.curve_fit(sinc2_func, theta, normalisedIntensity,p0=[1., 63.])


# Output results of fit to terminal
print('\n')
print('Results from fit:')
print(params)
print('\n')
print('Covariance matrix from fit:')
print(params_covariance)
print('\n')

# Draw a plot of "normalisedIntensity" as a function of "theta", with the fit on top.
plt.figure(figsize=(6, 4))
plt.scatter(theta, normalisedIntensity, label='Data',s=0.5)
plt.plot(theta, sinc2_func(theta, params[0], params[1]),label='Fitted function',color='red')
plt.legend(loc='best')
plt.show()
