import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import scipy.optimize as opt

num_of_peaks = 5#number of peaks to be fitted 
data_dir = "path/to/data/directory"#directory containing the data to be analysed. Please put all your data in this directory (and no other files!)


wavelengths = []#list to contain wavelength data for all spectra
intensities = []#list to contain intensity data for all spectra
file_names = []#list to contain the file names to keep track of which spectrum is which
'''
A 'deep copy' of the data is also contained within a list, intensities_dc. This prevents the actual data being overwritten later
'''
intensities_dc =[]

for file in os.listdir(data_dir):
    #this for loop reads in the data and assigns it to the lists created above
    data = np.genfromtxt(f"{data_dir}/{file}",delimiter=",",dtype=np.float64).T#read in the data
    '''
    appened the data read above onto the lists created above
    
    '''
    file_names.append(file)

def gauss(x,y0,mu,sigma,h):
    '''
    make this function return a normal distribution with:
    vertical offset y0
    mean value mu
    standard deviation sigma
    of height h
    '''
    return 0


#calculate the total number of files that will be processed
num_of_files = len(os.listdir(data_dir))

for i in range(num_of_files):
    #This for loop loops through each spectrum, plotting it and optimising for normal and lorentzian distributions
    
    #plot the measured data
    plt.scatter(wavelengths[i],intensities[i],marker='x',color="g",label = "measured data")

    for j in range(num_of_peaks):
        #This for loop loops through the 'num_of_peaks' largest peaks in the spectra measured, trying to fit peaks to them
        
        '''
        max_index stores the index of the highest point remaining in the distribution. intensities_dc is used here to keep track of which parts 
        of the spectrum have been fitted and which haven't
        '''
        max_index = np.argmax(intensities_dc[i])
        
        #low_cut is how far below the maximum the fitting algorithm will look. high_cut is equivalent for above
        low_cut = 200
        high_cut = 200
        

        #The if/elif statements check whether the approximate peak location is within 200 channels of the bottom/top of the distribution
        #and corrects if this is the case, preventing index errors later on
        if max_index<=200:
            low_cut = max_index
        elif max_index> np.shape(intensities[i]):
            high_cut = np.shape(intensities[i])-max_index

        #create the wavelength and intensity data subsets which are fitted by the scipy function
        x_subset =wavelengths[i][max_index-low_cut:max_index+high_cut]
        y_subset =intensities[i][max_index-low_cut:max_index+high_cut]

        #find a way to make reasonable guesses for the parameters of the peak being fitted. 
        #mu_guess is lalread done as an example
        
        mu_guess = wavelengths[i][max_index]
        y0_guess = 0
        sigma_guess = 0
        h_guess = 0

        #params and params_covariance store the normal distribution function's optimum values for each peak
        params = []
        params_covariance = []
        try:
            #attempt to fit the data
            params, params_covariance = opt.curve_fit(gauss,x_subset,y_subset,p0=[y0_guess,mu_guess,sigma_guess,h_guess])
        except RuntimeError:
            #if the peak could not be fitted (often because the fit is not "good enough"), a RuntimeErro is raised.
            # This exception catches this error
            print(f"Could not fit a peak for {file_names[i]}")
        else:
            #if the peak has been successfully fit, plot the data
            if j==0:
                plt.plot(x_subset,gauss(x_subset,params[0],params[1],params[2],params[3]),label =f"Normal distributions",color='r')
            else:
                plt.plot(x_subset,gauss(x_subset,params[0],params[1],params[2],params[3]),color='r')

            print('\nGaussian:')
            print(f"y0 estimated to be {params[0]} +/- {params_covariance[0][0]}")
            '''
            print out the rest of the optimum values found by scipy            
            '''

        for k in range(low_cut+high_cut):
            #this for loop sets all values in our copy of the intensities list to the minimum value in the array.
            #This stops this peaks being fitted again.
            intensities_dc[i][k+max_index-low_cut] = np.min(intensities_dc)

    #plot the data and include axis labeles and a title for the graph