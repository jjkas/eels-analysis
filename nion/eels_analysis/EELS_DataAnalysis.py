"""
    EELS Data Analysis

    A library of functions for EELS data analysis.
"""

# third party libraries
import numpy
import scipy.signal
import math


# local libraries
from . import CurveFittingAndAnalysis
from . import EELS_CrossSections

__experimental_edge_data = (None, None)

def find_experimental_edge_energies(eels_spectrum: numpy.ndarray, energy_range_ev: numpy.ndarray,
                                    search_range_ev: numpy.ndarray = None, **kwargs) -> numpy.ndarray:
    """Find energies where edges are located in experimental eels spectrum.

    Input:
        eels_spectrum    - array of spectral data
        energy_range_ev  - 2 element array containing energy range. First element is energy associated with start of
                           spectral element, second element is end of last spectral element, i.e. = # of elements * energy_step 
        search_range_ev  - 2 element array containing energy range to include in search for edges.
        **kwargs         - Options for controling the algorithm

    Returns:
        experimental_edge_energies_ev - array of edge energies.
    """
    global __experimental_edge_data

    # Set sensitivity parameter - used for general control of sensitivity of the algorithm to step like features.
    # Lower sensitivity will cause the algorithm to find fewer edges, while larger sensitivity will cause it to
    # find more edges. Ensure sensitivity is between 0 and 1. Default is 0.5.
    sensitivity_parameter = 1.0 - max(min(kwargs.get('sensitivity',0.5),1.0),0.0)

    # For debugging.
    debug_plotting = kwargs.get('debug_plotting',False)
    
    if search_range_ev is None:
        # Default low loss range is 0 to 30eV
        search_range_ev = numpy.array([max(energy_range_ev[0],30.0), energy_range_ev[-1]])
    else:
        # Check that search range contains a non-zero portion of energy range.
        assert (search_range_ev[-1] > energy_range_ev[0]) and (search_range_ev[0] < energy_range_ev[-1])
        
        # Set search range to allowable values.
        search_range_ev = numpy.array([max(search_range_ev[0],energy_range_ev[0]), min(search_range_ev[1],energy_range_ev[-1])])

    # Sanity checks:
    assert energy_range_ev[0] < energy_range_ev[1]
    assert search_range_ev[0] < search_range_ev[1]
    
    # Calulate energy step
    energy_step = (energy_range_ev[1] - energy_range_ev[0])/eels_spectrum.shape[-1]
    
    emin_search = search_range_ev[0]
    
    # Set maximum value of search range. Use max of spectrum if nothing is passed by user.
    emax_search = search_range_ev[1]
        
    # Find closest indices corresponding to emin_search and emax_search
    imin_search = int((emin_search - energy_range_ev[0])/energy_step)
    imax_search = int((emax_search - energy_range_ev[0])/energy_step)
    emin_search = energy_range_ev[0] + float(imin_search)*energy_step
    emax_search = energy_range_ev[0] + float(imax_search)*energy_step 
    
    
    # Create cut energy and spectral arrays.
    if imax_search >= eels_spectrum.shape[-1]-1:
        eels_spectrum_search = eels_spectrum[imin_search:]
    else:
        eels_spectrum_search = eels_spectrum[imin_search:imax_search]
        
    energies_search = numpy.linspace(emin_search,emax_search,eels_spectrum_search.shape[-1])
    assert eels_spectrum_search.shape[-1] == energies_search.shape[-1]

    if kwargs.get('re_analyze',True):
        
        # Here we will search for steplike features by creating an array of correlation coefficients correlating the energy axis
        # with the spectrum. Background signal will generally be negatively correlated with energy, while strong edges will be
        # positively correlated with energy, and weak edges will just show an increase in correlation. This method is much less dependent
        # on the size of the jump than the derivative method, so that small edges can be found.

        # Create space for correlation array
        corr_coeff_array = numpy.zeros(eels_spectrum_search.shape[-1])
            
        # Set energy range for calculating the correlation coefficient. A smaller energy range will be more
        # susceptible to noise, while a larger energy range will only be sensitive to changes the persist over
        # a larger energy range. Default to the min of 25eV or 1/10th the search range. 
        correlation_energy_range = min(kwargs.get("correlation_energy_range_ev",max(50.0*sensitivity_parameter,5*energy_step)),(emax_search-emin_search)/10.0)
        i_start = -1
        i_end = -1
        for i_energy, energy in enumerate(energies_search):
            # Create correlation coefficient between subarrays corresponding to current energy point to that minus correlation_energy_range.
            # This tends to give results which have a max derivative close to the edge.
            # coefficient is between -1 and 1, but we'll add 1 and devide by 2 to get a parameter that ranges from 0 to 1.
            # Ignore end points of energy and spectrum array (set to zero).
            energy_range_mask = numpy.logical_and(abs(energies_search - energies_search[i_energy]) <= correlation_energy_range, energies_search < energies_search[i_energy])
            spectrum_sub_array = eels_spectrum_search[numpy.where(energy_range_mask)]        
            energy_search_sub_array = energies_search[numpy.where(energy_range_mask)]
            
            # Set edges (beyond correlation energy range) to same as first valid point.
            if (energies_search[i_energy] - energies_search[0]) > correlation_energy_range and (energies_search[-1] - energies_search[i_energy]) > correlation_energy_range:
                corr_coeff_array[i_energy] = (numpy.corrcoef(energy_search_sub_array,spectrum_sub_array)[0,1] + 1.0)/2.0
                # Save the energy of the first valid point
                if i_start < 0:
                    i_start = i_energy

        # Set edge points to same as first valid point.
        corr_coeff_array[0:i_start-1] = corr_coeff_array[i_start]
    
        # Take first derivative of spectrum. 
        first_derivative = numpy.gradient(eels_spectrum_search,energies_search)
    
        # Now smooth first derivative via simple local averaging, then subtract off longer range local average.
        # Right now these are hard coded number of points average, but they should be set by keyword arguments
        # and defaults.
        num_avg=kwargs.get('derivative_smoothing',4)
    
        # Also average over a longer range only extending below the current point. This will give some measure of the average derivative of the background.
        num_lr_avg=num_avg*4
        first_derivative_avg = numpy.zeros(first_derivative.shape[-1])
        first_derivative_long_range_avg = numpy.zeros(first_derivative.shape[-1])

        for i,fd in enumerate(first_derivative):
            if (i-num_avg >= 0) and (i+num_avg <= first_derivative.shape[-1] - 1):
                first_derivative_avg[i] = numpy.average(first_derivative[i-num_avg:i+num_avg])
            else:
                first_derivative_avg[i] = numpy.average(first_derivative[0:num_avg])
                
                if (i-num_lr_avg >= 0): 
                    first_derivative_long_range_avg[i] = numpy.average(first_derivative[i-num_lr_avg:i])
                elif (i-num_lr_avg < 0):
                    first_derivative_long_range_avg[i] = numpy.average(first_derivative[0:num_lr_avg]) 
                else:
                    first_derivative_long_range_avg[i] = numpy.sum(first_derivative[i-num_lr_avg:])/(2*num_lr_avg+1)
            
            # Subtract the background derivative from the derivative
            first_derivative_avg[i] = (first_derivative_avg[i] - first_derivative_long_range_avg[i])/numpy.sqrt(eels_spectrum_search[i]) if eels_spectrum_search[i]>0 else 0.0 
        


        first_derivative=first_derivative_avg
    
        if False: # Meant for debugging only.
            plt.plot(energies_search, first_derivative, label='1st')
            plt.plot(energies_search, corr_coeff_array, label='corr')
            plt.legend(loc="upper left")
            plt.show()
        

        # Gain change range: Not implemented yet.
        gain_change_range = kwargs.get('gain_change_range_ev',numpy.array([0.0,0.0]))

        # Find indices of maxima in the derivative of spectrum.
        ind_max_first = scipy.signal.argrelextrema(first_derivative,numpy.greater)

        # Find indices of maxima in the correlation coefficient array.
        ind_max_corr = scipy.signal.argrelextrema(corr_coeff_array,numpy.greater)

        # Find regions with possible edges, based on region from max in correlation coefficient to that minus
        # correlation energy range. For all of these, we will assign an edge energy, and a "quality factor" based on
        # the size of the maximum derivative in the region, and the size of the correlation coefficient at it's max.
        q_factors2=numpy.array([])
        edge_energies2 = numpy.array([])
        # For each maximum in correlation coefficient array, check if there is also a maximum
        # in the first derivative between the energy of the maximum and the energy of the previous maximum in the correlation
        # coefficient.

        for ind in ind_max_corr[0]:
            if ind != ind_max_corr[0][0]:
                # If this is not the first max, set emin and emax to define energy range to look for maxima in the
                # first derivative.
                # emin is the energy of the previous maximum in the correlation coefficient.
                emin = emax
                # emax is the energy of the current maximum in the correlation coefficient.
                emax = energies_search[ind]
            else:
                # If this is the first max, set emin to emax - correlation_energy_range. 
                emax = energies_search[ind]
                emin = emax - correlation_energy_range

            # Define an array of the maxima of the first derivatives in the region, along with its corresponding energies.
            first_derivative_in_region = first_derivative[ind_max_first][numpy.where(abs(energies_search[ind_max_first] - (emin+emax)/2.0) <= (emax - emin)/2.0)]
            energies_in_region = energies_search[ind_max_first][numpy.where(abs(energies_search[ind_max_first] - (emin+emax)/2.0) <= (emax - emin)/2.0)]
            
            # If there are no maxima in the first derivative in this region, discount this point
            if first_derivative_in_region.size > 0:
                # Find the index of the maximum in first derivative in the region, and the corresponding maximum value.
                ind_max_first_in_region = numpy.argmax(first_derivative_in_region)
                max_first_in_region = first_derivative_in_region[ind_max_first_in_region]

                # Define the edge energy as that where the maximum in first derivative occurs.
                edge_energy = energies_in_region[ind_max_first_in_region]

                if True:
                    # Define a quality factor equal to max in first derivative times max in correlation coefficient. 
                    if (edge_energies2.size == 0):
                        edge_energies2 = numpy.append(edge_energies2,edge_energy)
                        q_factors2 = numpy.append(q_factors2, max_first_in_region*corr_coeff_array[ind])
                    else:
                        if edge_energies2[-1] != edge_energy:
                            edge_energies2 = numpy.append(edge_energies2,edge_energy)
                            q_factors2 = numpy.append(q_factors2, max_first_in_region*corr_coeff_array[ind])
                        else:
                            q_factors2[-1] = max(max_first_in_region*corr_coeff_array[ind],q_factors2[-1])

    else:
        q_factors2 = __experimental_edge_data[1]
        edge_energies2 = __experimental_edge_data[0]
        
    # Parameters to control filtering
    # Below this is filtering options
    
    # We will take only first edge found after each maximum in the corr_coeff_array - edge_separation_energy.
    # Default is 7.5eV, minimum is 2*energy step. Multiply edge_separation parameter by edge energy.
    edge_separation_energy = kwargs.get('edge_separation_parameter',10.0*sensitivity_parameter/100.0)

    # Define a deviation. This isn't perfect, but it's close enough. Most edges are way above the median deviation.
    
    median = numpy.median(q_factors2)
    sigma0 = scipy.stats.median_absolute_deviation(q_factors2)
    sigma = scipy.stats.median_absolute_deviation(q_factors2[abs(q_factors2-median) < 3.0*sigma0])
    cut= median + sigma*7.0
    cutbig= 0.0 #cut #median + sigma*20.0
    #edge_energies_final = numpy.array([])
    #q_factors_final = numpy.array([])
    q_factors_major = numpy.array([])
    q_factors_minor = numpy.array([])
    edge_energies_major = numpy.array([])
    edge_energies_minor = numpy.array([])
    #part_of_last_edge = False
    denom = sigma
    max_qf = max(q_factors2)
    correlation_cutoff_scale = max_qf*max(min(kwargs.get('correlation_cutoff_scale',sensitivity_parameter),1.0),0.0)**6/denom

    min_corr_major = correlation_cutoff_scale
    min_corr_minor = 0.0 #min_corr_major/5.0
    for ind,qfac in enumerate(q_factors2):
        # get standard deviation of nearby q_factors to normalize quality factor
        #denom=numpy.std(numpy.abs(q_factors2[max(ind-5,0):max(ind-1,2)]))
        #denom = 1.0
        if (qfac/denom > min_corr_major):
            if edge_energies_major.shape[-1] > 0:
                if (edge_energies2[ind] - edge_energies_major[-1] > edge_separation_energy*edge_energies2[ind]): 
                    q_factors_major = numpy.append(q_factors_major,qfac/denom)
                    edge_energies_major = numpy.append(edge_energies_major,edge_energies2[ind])
            else:
                q_factors_major = numpy.append(q_factors_major,qfac/denom)
                edge_energies_major = numpy.append(edge_energies_major,edge_energies2[ind])
            
        elif (qfac/denom > min_corr_minor):  #and (qfac >= q_factors2[max(ind-1,0)]) and (qfac >= q_factors2[min(ind+1,q_factors2.shape[-1]-1)]):
            if edge_energies_minor.shape[-1] > 0:
                if (edge_energies2[ind] - edge_energies_minor[-1] > edge_separation_energy*edge_energies2[ind]): 
                    q_factors_minor = numpy.append(q_factors_minor,qfac/denom)
                    edge_energies_minor = numpy.append(edge_energies_minor,edge_energies2[ind])
            else:
                q_factors_minor = numpy.append(q_factors_minor,qfac/denom)
                edge_energies_minor = numpy.append(edge_energies_minor,edge_energies2[ind])

    if debug_plotting: # For debugging purposes.
        import matplotlib.pyplot as plt 
        plt.stem(edge_energies2,q_factors2,linefmt='C2-',markerfmt='o',label='qf2')
        plt.yscale('log')
        if q_factors_major.size >= 1:
            plt.stem(edge_energies_major,q_factors_major, linefmt='C0-',markerfmt='x', label='major edges')
        if q_factors_minor.size >= 1:
            plt.stem(edge_energies_minor,q_factors_minor, linefmt='C1-',markerfmt='s', label='minor edges')
        plt.plot(energies_search, eels_spectrum_search/numpy.amax(eels_spectrum_search))
        plt.legend(loc="upper right")
        plt.show()

    # Return edge energies and q_factors sorted by q_factor from largest to smallest.
    ind_sort = numpy.argsort(-q_factors_major)

    # Update global data. We can use this to refilter without re-analyzing the spectrum.
    __experimental_edge_data = (edge_energies2, q_factors2)
    return edge_energies_major[ind_sort], q_factors_major[ind_sort]

def zero_loss_peak(low_loss_spectra: numpy.ndarray, low_loss_range_eV: numpy.ndarray) -> tuple:
    """Isolate the zero-loss peak from low-loss spectra and return the zero-loss count, zero-loss peak, and loss-spectrum arrays.

    Returns:
        zero_loss_counts - integrated zero-loss count array
        zero_loss_peak - isolated zero-loss peak spectral array
        loss_spectrum - residual loss spectrum array
    """
    pass

def core_loss_edge(core_loss_spectra: numpy.ndarray, core_loss_range_eV: numpy.ndarray, edge_onset_eV: float, edge_delta_eV: float,
                    background_ranges_eV: numpy.ndarray, background_model_ID: int = 0) -> tuple:
    """Isolate an edge signal from background in core-loss spectra and return the edge integral, edge profile, and background arrays.

    Returns:
        edge_integral - array of integrated edge counts evaluated over the delta window past the edge onset
        edge_profile - array of isolated edge profiles evaluated over the profile range (see below)
        edge_background - array of background models evaluated over the profile range (see below)
        profile_range - contiguous union of edge delta and background ranges
    """
    edge_onset_margin_eV = 0
    assert edge_onset_eV > core_loss_range_eV[0] + edge_onset_margin_eV

    edge_range = numpy.full_like(core_loss_range_eV, edge_onset_eV)
    edge_range[0] -= edge_onset_margin_eV
    edge_range[1] += edge_delta_eV
    poly_order = 1
    fit_log_y = (background_model_ID <= 1)
    fit_log_x = (background_model_ID == 0)

    return CurveFittingAndAnalysis.signal_from_polynomial_background(core_loss_spectra, core_loss_range_eV, edge_range,
                                                                        background_ranges_eV, poly_order, fit_log_y, fit_log_x)

def relative_atomic_abundance(core_loss_spectra: numpy.ndarray, core_loss_range_eV: numpy.ndarray, background_ranges_eV: numpy.ndarray,
                                atomic_number: int, edge_onset_eV: float, edge_delta_eV: float,
                                beam_energy_eV: float, convergence_angle_rad: float, collection_angle_rad: float) -> numpy.ndarray:
    """Isolate the specified edge signal from the core-loss spectra and compute a relative atomic concentration value.

    Returns:
        atomic_abundance - integrated edge counts divided by the partial cross-section over the delta range,
        in units of (spectrum counts) * atoms / (nm * nm).
    """
    edge_data = core_loss_edge(core_loss_spectra, core_loss_range_eV, edge_onset_eV, edge_delta_eV, background_ranges_eV)

    # The following should ultimately be pulled out of the edge ID table, based on atomic number and edge onset
    shell_number = 1
    subshell_index = 1
    cross_section = EELS_CrossSections.partial_cross_section_nm2(atomic_number, shell_number, subshell_index, edge_onset_eV, edge_delta_eV,
                                                                    beam_energy_eV, convergence_angle_rad, collection_angle_rad)
    atomic_abundance = edge_data[0] / cross_section
    return atomic_abundance

def atomic_areal_density_nm2(core_loss_spectra: numpy.ndarray, core_loss_range_eV: numpy.ndarray, background_ranges_eV: numpy.ndarray,
                                low_loss_spectra: numpy.ndarray, low_loss_range_eV: numpy.ndarray,
                                atomic_number: int, edge_onset_eV: float, edge_delta_eV: float,
                                beam_energy_eV: float, convergence_angle_rad: float, collection_angle_rad: float) -> numpy.ndarray:
    """Isolate the specified edge signal from the core-loss spectra and compute the implied atomic areal density.

    Returns:
        atomic_areal_density - edge counts divided by the low-loss intensity and partial cross-section, integrated over the delta range,
        in atoms / (nm * nm).
    """
    pass
