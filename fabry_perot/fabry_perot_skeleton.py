#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd
import os


# User-input parameters
# You should work these out by looking at the data and identifying a region of
# peaks that you can attribute to the etalon.
# This may be different for different spectrum measurements.
peak_minimum_height = 0.05  # The minimum height for a peak to be considered
peak_minimum_width = 10  # The minimum width for a peak to be considered [nm]


# Define a Gaussian function, which we will use later to fit to etalon
# transmisson peaks
def gaussian(x, mu, sig, h, y0):
    exit('You need to define a Gaussian function')  # Remove this
    return  # Define your Gaussian here


# Read spectrum data from csv file
# You should put all the files to analyse in a folder with nothing else in
# it, and provide the path to that folder below.
data_dir = r'path/to/data'
for file_name in os.listdir(data_dir):
    df_etalon = pd.read_csv(os.path.join(data_dir, file_name))

    # Extract wavelength and intensity values from the pandas dataframe and
    # subtract background reference
    wavelength = np.array(df_etalon.iloc[:, 0])
    intensity = np.array(df_etalon.iloc[:, 1])

    # Find peaks in spectrum corresponding to constructive interference
    # in the etalon
    peaks, peaks_properties = find_peaks(intensity,
                                         width=peak_minimum_width,
                                         height=peak_minimum_height)

    # Find first and last peaks and set lower and upper limits for plot
    first_peak, last_peak = peaks[0], peaks[-1]
    lower_limit = first_peak - peak_minimum_width
    upper_limit = last_peak + peak_minimum_width

    plt.scatter(wavelength[lower_limit:upper_limit],  # x data
                intensity[lower_limit:upper_limit],  # y data
                s=0.5)  # datapoint size in pt

    # Create empty lists to fill with the lambda_0, fsr and fwhm values
    lambda_0 = []
    free_spectral_range = []
    full_width_half_maximum = []

    # Iterate over the peaks we identified to determine lambda_0, fsr and fwhm
    for i in range(len(peaks) - 1):
        # Set current and next peak location
        peak = peaks[i]
        next_peak = peaks[i + 1]

        # Find midpoint and separation of each adjacent pair of peaks
        midpoint = (wavelength[peak] + wavelength[next_peak]) / 2
        separation = wavelength[next_peak] - wavelength[peak]

        # Find the separation beteen the two peaks in terms of data points
        dp_separation = next_peak - peak

        # Calculate lower and upper limit to use for peak fitting
        peak_lower_limit = int(peak - dp_separation/2)
        peak_upper_limit = int(peak + dp_separation/2)

        # Slice the wavelength and intensity data to obtain a region around the
        # peak, to which we will try to fit a Gaussian curve
        peak_x = wavelength[peak_lower_limit:peak_upper_limit]
        peak_y = intensity[peak_lower_limit:peak_upper_limit]

        # Append values to lists
        lambda_0.append(midpoint)
        free_spectral_range.append(separation)

        # Fit the Gaussian to the region around the peak
        params, params_covariance = curve_fit(gaussian,  # The function to fit
                                              peak_x,  # The x data
                                              peak_y,  # The y data
                                              p0=[wavelength[peak],
                                                  6.,
                                                  intensity[peak],
                                                  min(peak_y),
                                                  0.],  # Initial parameters
                                              maxfev=10000)  # Max iterations

        # Calculate FWHM from the standard deviation of the Gaussian fit, and
        # append it to a list
        fwhm = 2 * np.sqrt(2 * np.log(2)) * params[1]
        full_width_half_maximum.append(fwhm)

        # Plot the fit for each peak
        x = np.linspace(min(peak_x), max(peak_x), 100)
        plt.plot(x, gaussian(x, *params), c='r')

    # We now construct a data object which we can save to a file
    peaks_data = {'lambda_0': lambda_0,
                  'fsr': free_spectral_range,
                  'fwhm': full_width_half_maximum}

    # Use pandas to write the data object to a csv file
    os.makedirs('output')
    df_output = pd.DataFrame(peaks_data)
    df_output.to_csv('output/characterised_peaks_{}.csv'.format(file_name))
    plt.show()
