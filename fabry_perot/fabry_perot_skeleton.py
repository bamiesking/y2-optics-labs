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


# Define a function of the form of a Gaussian plus a straight line (mx + c),
# which we will use later to fit to transmisson peaks.
def fit_func(x, mu, sigma, h, offset, slope):
    exit('You need to define a fit function')  # Remove this
    return mu*x + sigma  # Define your fitting function.


# Read spectrum data from csv file.
# You should put all the files to analyse in a folder with nothing else in
# it, and provide the path to that folder below.
data_dir = r'fabry_perot/data'
for file_name in os.listdir(data_dir):

    # Make sure the file is not hidden (such as .DS_Store on macOS).
    if file_name[0] is ".":
        continue  # This skips the current iteration of the loop.

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

    if len(peaks) == 0:
        print('No peaks found in {file}'.format(file=file_name))
        continue

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

        # Find midpoint and separation of each adjacent pair of peaks. Think
        # about these - is there a better way of calculating either of these
        # values?
        midpoint = (wavelength[peak] + wavelength[next_peak]) / 2
        separation = wavelength[next_peak] - wavelength[peak]

        # Find the separation beteen the two peaks in terms of data points
        dp_separation = next_peak - peak

        # Calculate lower and upper limit to use for peak fitting
        peak_lower_limit = int(peak - dp_separation/2)
        peak_upper_limit = int(peak + dp_separation/2)

        # Slice the wavelength and intensity data to obtain a region around the
        # peak, to which we will try to fit the function we defined earlier.
        peak_x = wavelength[peak_lower_limit:peak_upper_limit]
        peak_y = intensity[peak_lower_limit:peak_upper_limit]

        # Make calculations for initial guesses. Think about these values and
        # why they comprise reasonable initial values.
        initial_mu = wavelength[peak]
        initial_sigma = (max(peak_x) - min(peak_x))/2
        initial_h = (max(peak_y) - min(peak_y))
        initial_offset = min(peak_y)
        initial_slope = peak_y[-1] - peak_y[0]

        # Pack all the initial guesses together into a list which we will pass
        # to the cirve_fit function. You should adjust them here in order to
        # find the correct minimisation.
        initial_guesses = [
            initial_mu,
            initial_sigma,
            initial_h,
            initial_offset,
            initial_slope
        ]

        try:
            # Fit the Gaussian to the region around the peak
            params, params_covariance = curve_fit(
                fit_func,  # The function to fit
                peak_x,  # The x data
                peak_y,  # The y data
                p0=initial_guesses,  # Initial parameters
                maxfev=10000  # Max iterations
            )
        except Exception:
            # If the fit fails, print out a message.
            print('curve_fit failed for curve {} \
                   in file {}'.format(i, file_name))
            continue

        # Append values to lists
        lambda_0.append(midpoint)
        free_spectral_range.append(separation)

        # Calculate FWHM from the standard deviation of the Gaussian fit, and
        # append it to a list
        fwhm = 2 * np.sqrt(2 * np.log(2)) * params[1]
        full_width_half_maximum.append(fwhm)

        # Plot the fit for each peak
        x = np.linspace(min(peak_x), max(peak_x), 100)
        plt.plot(x, fit_func(x, *params), c='r')

    # We now construct a data object which we can save to a file
    peaks_data = {'lambda_0': lambda_0,
                  'fsr': free_spectral_range,
                  'fwhm': full_width_half_maximum}

    # Use pandas to write the data object to a csv file
    try:
        # Try to make a folder to store the output in
        os.makedirs('output')
    except FileExistsError:
        # If the folder already exists, just skip
        pass
    df_output = pd.DataFrame(peaks_data)
    df_output.to_csv('output/characterised_peaks_{}'.format(file_name))
    plt.show()
