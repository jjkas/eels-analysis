"""
    EELS Data Analysis

    A library of functions for EELS data analysis.
"""

# third party libraries
import numpy
import scipy.signal
import math
# The following is a library that allows range dictionaries through the RangeDict object.
import typing
import copy

# local libraries
from nion.eels_analysis import CurveFittingAndAnalysis
from nion.eels_analysis import eels_analysis 
from nion.eels_analysis import PeriodicTable

__experimental_edge_data = (None, None)

def range_dict(begin, end, value):
    ''' Returns a dictionary that has keys in the range from begin - end, where begin and end are integers. The value of all items is set to value. '''
    d = {}
    for i in range(begin,end+1):
        d.update({i: value})

    return d

def find_species_from_experimental_edge_data(eels_spectrum: numpy.ndarray, energy_range_ev: numpy.ndarray, experimental_edge_data, search_range_ev, **kwargs) -> list:
    """Find chemical species associated with edge energy data produced by find_experimental_edge_energies along with spectrum.
       
    Input:
        eels_spectrum    - array of spectral data
        energy_range_ev  - 2 element array containing energy range. First element is energy associated with start of
                           first spectral element, second element is end of last spectral element, i.e. = # of elements * energy_step + energy_range_ev[0]
        experimental_edge_data - 2 element tuple that contains numpy array of edge_energies and array of quality factor (quality factor indicates how lilkely it is 
                                     that this is a real edge.
        **kwargs         - Options for controling the algorithm
            only_major_edges - Boolean: If True, do not include minor edges in search. Default -> True
            element_list - List of atomic numbers of elements to include in the search. If not supplied, all elements in periodic table (at present listed
                                 up to U (92) are included.
    
    Output:
        atom_data - list of lists: 
                          atom_data[i][0] - int: Atomic number of elements possibly associated with this experimental spectrum.
                          atom_data[i,1] - list of tuples, each containting (edge_name, number_of_edges_for_this_element, theoretical_edge_energy, experimental_edge_energy) 
                                           Note that each experimental edge is associated with only one theoretical edge per element and vice versa, i.e., there is a one-to-one
                                           correspondence between theoretical_edge <-> experimental_edge per element. 
    """
    # Sanity checks
    # Make sure eels_spectrum holds data (probably should be > 10 or 20 or something, but I'll put 1 for now.
    assert eels_spectrum.shape[-1] > 1

    # Check energy range is sensible
    assert energy_range_ev[1] > energy_range_ev[0]
    assert energy_range_ev[1] > 0
    assert search_range_ev[1] > search_range_ev[0]
    assert search_range_ev[0] > 0
    search_range_ev[0] = max(search_range_ev[0],energy_range_ev[0])
    search_range_ev[1] = min(search_range_ev[1],energy_range_ev[1])
    
    # Check that experimental edge data is sensible and has the right structure. experimental edge data should be tuple that holds
    # (edge_energies, quality_factor).
    assert len(experimental_edge_data) == 2

    # Check that the tuple holds arrays with the same shape.
    assert experimental_edge_data[0].shape == experimental_edge_data[1].shape

    # Check that experimental edge energies are within the energy range (maybe we want to allow for any rather than forcing all)
    assert numpy.all(experimental_edge_data[0] >= energy_range_ev[0])
    assert numpy.all(experimental_edge_data[0] <= energy_range_ev[1])

    only_major_edges = kwargs.get('only_major_edges',True)
    element_list = kwargs.get('element_list', [])
    # Designation of major edges - only including up to 5000 eV.
    #major_edges = ['K','L2','L23','L3','M4','M45','M5','N4','N45','N5','N6','N67','N7','O4','O45','O5']
    # Reduced only includes first of series, e.g., L3 if L2,L3.
    major_edges_reduced = {}
    major_edges_reduced.update(range_dict(1,9,('K')))
    major_edges_reduced.update(range_dict(10,17,('K','L23','L3')))
    major_edges_reduced.update(range_dict(18,22,('K','L23','L3','M23','M3')))
    major_edges_reduced.update(range_dict(23,28,('L23','L3','M23','M3')))
    major_edges_reduced.update(range_dict(29,31,('L23','L3')))
    major_edges_reduced.update(range_dict(32,35,('L23','L3','M45','M5')))
    major_edges_reduced.update(range_dict(36,46,('L23','L3','M45','M5','N23','N3')))
    major_edges_reduced.update(range_dict(47,49,('L23','L3','M45','M5')))
    major_edges_reduced.update(range_dict(50,53,('L23','L3','M45','M5','N45','N5')))
    major_edges_reduced.update(range_dict(54,55,('M45','M5','N45','N5')))
    major_edges_reduced.update(range_dict(56,71,('M45','M5','N45','N5','O23','O3')))
    major_edges_reduced.update(range_dict(72,77,('M45','M5','O23','O3')))
    major_edges_reduced.update(range_dict(78,81,('M45','M5')))
    major_edges_reduced.update(range_dict(82,83,('M45','M5','O45','O5')))
    major_edges_reduced.update(range_dict(84,84,('M45','M5')))
    major_edges_reduced.update(range_dict(90,92,('M45','M5','N67','N7','O45','O5')))

    # Start by looping through experimental edge energies and finding all tabulated electron shells that have edges near this.
    # Edge energies are ordered by quality factor from largest to smallest, so most important edges will be checked first.
    ptable = PeriodicTable.PeriodicTable()
    i_edge = 0
    element_data = {}
    for edge_energy in experimental_edge_data[0]:
        q_factor = experimental_edge_data[1][i_edge]
        # Set energy range to look for other edges. For now do +/- max of 3% or 10eV.
        max_energy_diff = max(0.03*edge_energy,10.0)
        energy_range = [edge_energy-max_energy_diff,edge_energy+max_energy_diff]
        # Create a dictionary of all elements that might be in this system.
        element_data.update(ptable.find_elements_in_energy_interval(energy_range))


    # We now have a list of elements that have an edge in the region associated with the edge found in the experiment. The element data
    # includes the atomic number, and a list of all edges (label, energy).
    # Now loop through elements and search for edges that match those in the experiment. Count how many matches there
    # are for each element, and store the difference between the energy for that element and that for the experimental edge.
    # Count only the closest edge for each element for each experimental edge.
    # Loop through atoms
    atom_data = []
    i_atom = 0
    for atom,edges in element_data.items():
        if (not element_list) or (int(atom) in element_list):
            # Get major edges for this element that lie within the range of the experimental data.
            major_edges_for_this_atom = []
            for edge_label,energy in edges.items():
                if int(atom) in major_edges_reduced:
                    if edge_label in major_edges_reduced[int(atom)] and (search_range_ev[0] < energy < search_range_ev[1]):
                        major_edges_for_this_atom = major_edges_for_this_atom + [edge_label]
                    
            # Loop through experimental edges, and find best one for this edge if it exists.
            # Number of edges matched will be one measure of the probability that an element exists in this spectrum.
            number_edges_matched = 0
            has_edge = False
            missing_major_edge = False
            for exp_edge_energy in experimental_edge_data[0]:
                # Loop through edges for this atom. For each experimental edge, there should
                # be a max of one match per atom.
                min_diff = max(exp_edge_energy*0.03,15.0) 
                best_edge_name = None
                for edge_name,energy in edges.items():
                    if (not only_major_edges) or (edge_name in major_edges_for_this_atom):
                        diff = numpy.absolute(energy-exp_edge_energy)
                        if diff <= min_diff:
                            best_edge_for_this_exp_edge = energy
                            best_edge_name = edge_name
                            min_diff = diff
                            has_edge = True

                if best_edge_name:
                    number_edges_matched += 1
                    if len(atom_data) - 1 < i_atom:
                        atom_data = atom_data + [[int(atom)]]
                        atom_data[i_atom] = atom_data[i_atom] + [[(best_edge_name,best_edge_for_this_exp_edge,exp_edge_energy)]]
                    else:
                        atom_data[i_atom][1] = atom_data[i_atom][1] + [(best_edge_name,best_edge_for_this_exp_edge,exp_edge_energy)]
                    
                    if best_edge_name in major_edges_for_this_atom:
                        major_edges_for_this_atom.remove(best_edge_name)

            if has_edge:
                if True: #if len(major_edges_for_this_atom) == 0:
                    #atom_data[i_atom].insert(1,number_edges_matched)
                    i_atom += 1
                else:
                    del atom_data[i_atom]

    return atom_data
    
    
def find_experimental_edge_energies(eels_spectrum: numpy.ndarray, energy_range_ev: numpy.ndarray,
                                    search_range_ev: numpy.ndarray = None, **kwargs) -> numpy.ndarray:
    """Find energies where edges are located in experimental eels spectrum. In particular find an exhaustive list of edges and quality factors associated with
       features that might be edges. Then filter based on size of quality factor. 

    Input:
        eels_spectrum    - array of spectral data
        energy_range_ev  - 2 element array containing energy range. First element is energy associated with start of
                           first spectral element, second element is end of last spectral element, i.e. = # of elements * energy_step + energy_range_ev[0]
        search_range_ev  - 2 element array containing energy range to include in search for edges.

        **kwargs         - Options for controling the algorithm
                   sensitivity: Should be between 0 and 1, although this is enforced below. Sets overall sensitivity of edge finder, 0 returns only one edge,
                                1 returns (too) many edges, likely showing experimental noise etc.   
                   debug_plotting: Plots the spectrum and quality factors for debugging and analysis. For developers only.
                   reanalyze: If True (default) perform full analysis of experimental spectrum to find experimental edges. If False, do not re-analyze experimental
                              data (don't find edges or quality factors), but start with the exhaustive list of edges, quality factors from previous full call, and 
                              only perform the filtering step. 
                   correlation_energy_range_ev: Controls the smooting of the correlation coefficient array. Larger range, less sensitivity to noise, but may miss edges.
                              Smaller range is susceptible to noice. 
                              Default: sensitivity*50eV (default sensitivity is 0.5, so 25eV)
                                       Note: If correlation_energy_range_ev is larger than the width of search_range_ev/10, then it will be set to search_range_ev/10
                                             Subsequently, if correlation_energy_range_ev is smaller than 5*energy_step_ev (calculated below) it will be set to 5*energy_step_ev.
                                             These checks are important, and keep the range from being set too small or too large.
                              Explanation:
                              As part of the analysis, an array of correlation coefficients are calculated, to represent the correllation of the 
                              energy axis and the spectrum axis at each point. The ith correlation coefficient is calculated between an array of energies 
                              that extends from energy[i]-correlation_energy_range_ev to energy[i], and the corresponding subarray of the spectrum. 
                   derivative_smoothing_npoints: Number of points to use in derivative smoothing. Simple running average used, averaging the derivative of the spectrum
                              between [i-derivative_smoothing_npoints:i+derivative_smoothing_npoints]
                              Note: If derivative_smoothing_npoints > number of points in search range / 10, use number of points in search range / 10.
                   edge_separation_parameter: Ignore any edges found between current edge and edge*(1+edge_separation_parameter). Default is sensitivity/50 (0.01 if sensitivity
                              is left at 0.5). 
                              Note: There are hard coded limits that 10eV < edge*edge_separation_parameter < 50eV. 
                   cutoff_scale: Sets cutoff of quality factor for edge determination. If 0, all edges will be included, if large, very few will be included.
                              Note: There is a hard-coded min of 0, and a max that will give one edge for the spectrum.
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
        assert (search_range_ev[1] > energy_range_ev[0]) and (search_range_ev[0] < energy_range_ev[1])
        
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
        correlation_energy_range = max(min(kwargs.get("correlation_energy_range_ev",50.0*sensitivity_parameter),(emax_search-emin_search)/10.0), 5*energy_step)
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
        num_avg=max(min(kwargs.get('derivative_smoothing_npoints',2),first_derivative.shape[-1]),0)
    
        # Also average over a longer range only extending below the current point. This will give some measure of the average derivative of the background.
        num_lr_avg=min(max(num_avg*2,16),int(first_derivative.shape[-1]/5))
        cumsum_vec = numpy.cumsum(first_derivative)
        if num_avg > 0:
            first_derivative_avg = numpy.zeros(first_derivative.shape[-1])
            first_derivative_avg = (cumsum_vec[num_avg:] - cumsum_vec[:-num_avg]) / num_avg
        else:
            first_derivative_avg[:] = first_derivative[:]

        first_derivative_long_range_avg = numpy.zeros(first_derivative.shape[-1])
        first_derivative_long_range_avg = (cumsum_vec[num_lr_avg:] - cumsum_vec[:-num_lr_avg]) / num_lr_avg

        first_derivative2 = first_derivative[:]
        first_derivative2[int(num_lr_avg/2):-int(num_lr_avg/2)]=(first_derivative_avg[int((num_lr_avg-num_avg)/2):-int((num_lr_avg-num_avg)/2)] - first_derivative_long_range_avg)
        first_derivative2[0:int(num_lr_avg/2-1)] = 0.0
        first_derivative2[-int(num_lr_avg/2+1):-1] = 0.0
        first_derivative2 = first_derivative2/numpy.sqrt(eels_spectrum_search)
        first_derivative = first_derivative2[:]
        
        if False: #debug_plotting: # Meant for debugging only.
            import matplotlib.pyplot as plt 
            plt.plot(energies_search, eels_spectrum_search/numpy.amax(eels_spectrum_search))
            plt.plot(energies_search, first_derivative, label='1st')
            plt.plot(energies_search, first_derivative2, label='1st 2')
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
                # emin is the larger of the previous edge energy, and emax - correlation_energy_range
                emin = max(emax,energies_search[ind]-correlation_energy_range)
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
                for ind_max_first_in_region,max_first_in_region in enumerate(first_derivative_in_region):
                    #max_first_in_region = first_derivative_in_region[ind_max_first_in_region]

                    # Define the edge energy as that where the maximum in first derivative occurs.
                    edge_energy = energies_in_region[ind_max_first_in_region]
                    if max_first_in_region > 0:
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

    if q_factors2.size == 0:
        return
    # Parameters to control filtering
    # Below this is filtering options
    
    # We will take only first edge found after each maximum in the corr_coeff_array - edge_separation_energy.
    # Multiply edge_separation parameter by edge energy.
    edge_separation_parameter = kwargs.get('edge_separation_parameter',sensitivity_parameter/50.0)

    # Define a deviation. This isn't perfect, but it's close enough. Most edges are way above the median deviation.
    
    median = numpy.median(q_factors2)
    sigma0 = scipy.stats.median_absolute_deviation(q_factors2)
    sigma = sigma0 #scipy.stats.median_absolute_deviation(q_factors2[abs(q_factors2-median) < 3.0*sigma0])
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
    q_factors2 = q_factors2/denom
    cutoff_scale = max(min(kwargs.get('cutoff_scale',sensitivity_parameter*2.0),max(q_factors2)/20.0),0.0)
    min_corr_major = cutoff_scale
    min_corr_minor = 0.0 #min_corr_major/5.0
    qfmax=10.0*cutoff_scale
    for ind,qfac in enumerate(q_factors2):
        # get standard deviation of nearby q_factors to normalize quality factor
        #denom=numpy.std(numpy.abs(q_factors2[max(ind-5,0):max(ind-1,2)]))
        #denom = 1.0
        if (qfac >= 6.0*cutoff_scale): #min_corr_major):
            if edge_energies_major.shape[-1] > 0:
                if (edge_energies2[ind] - edge_energies_major[-1] > min(max(edge_separation_parameter*edge_energies2[ind],10.0),50.0)):
                    qfm = qfac/numpy.average(q_factors2[max(ind-5,0):ind])
                    if qfm > qfmax or (qfac >= 20.0*cutoff_scale and qfm > 5.0*cutoff_scale):
                        #print(qfac,numpy.average(q_factors2[max(ind-3,0):ind]),qfm)
                        edge_energies_major = numpy.append(edge_energies_major,edge_energies2[ind])
                        q_factors_major = numpy.append(q_factors_major,qfm)                
            elif ind > 1:
                qfm = qfac/numpy.average(q_factors2[max(ind-5,0):ind])
                if qfm > qfmax or (qfac >= 20.0*cutoff_scale and qfm > 5.0*cutoff_scale):
                    edge_energies_major = numpy.append(edge_energies_major,edge_energies2[ind])
                    q_factors_major = numpy.append(q_factors_major,qfm)
            elif ind > 0:
                qfm = qfac/q_factors2[ind-1]
                if qfm > qfmax or (qfac >= 20.0*cutoff_scale and qfm > 5.0*cutoff_scale):
                    edge_energies_major = numpy.append(edge_energies_major,edge_energies2[ind])
                    q_factors_major = numpy.append(q_factors_major,qfm)                

            else:
                qfm = qfac
                if qfm > qfmax:
                    edge_energies_major = numpy.append(edge_energies_major,edge_energies2[ind])
                    q_factors_major = numpy.append(q_factors_major,qfm)                

        elif (qfac/denom > min_corr_minor):  #and (qfac >= q_factors2[max(ind-1,0)]) and (qfac >= q_factors2[min(ind+1,q_factors2.shape[-1]-1)]):
            if edge_energies_minor.shape[-1] > 0:
                if (edge_energies2[ind] - edge_energies_minor[-1] > edge_separation_parameter*edge_energies2[ind]): 
                    q_factors_minor = numpy.append(q_factors_minor,qfac/denom)
                    edge_energies_minor = numpy.append(edge_energies_minor,edge_energies2[ind])
            else:
                q_factors_minor = numpy.append(q_factors_minor,qfac/denom)
                edge_energies_minor = numpy.append(edge_energies_minor,edge_energies2[ind])

    if debug_plotting: # For debugging purposes.
        import matplotlib.pyplot as plt 
        plt.stem(edge_energies2,q_factors2,linefmt='C2-',markerfmt='o',label='qf2',use_line_collection=True)
        plt.yscale('log')
        if q_factors_major.size >= 1:
            plt.stem(edge_energies_major,q_factors_major, linefmt='C0-',markerfmt='x', label='major edges',use_line_collection=True)
        #if q_factors_minor.size >= 1:
            #plt.stem(edge_energies_minor,q_factors_minor, linefmt='C1-',markerfmt='s', label='minor edges')
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
                                beam_energy_eV: float, convergence_angle_rad: float, collection_angle_rad: float):
    """Isolate the specified edge signal from the core-loss spectra and compute a relative atomic concentration value.

    Returns:
        atomic_abundance - integrated edge counts divided by the partial cross-section over the delta range,
        in units of (spectrum counts) * atoms / (nm * nm).
        error            - combined experimental and theoretical error in atomic abundance.
    """
    edge_data = core_loss_edge(core_loss_spectra, core_loss_range_eV, edge_onset_eV, edge_delta_eV, background_ranges_eV)
    if False:
        print(edge_onset_eV,edge_delta_eV)
        print(edge_data[0])
        import matplotlib.pyplot as plt
        delta = (core_loss_range_eV[1]-core_loss_range_eV[0])/core_loss_spectra.shape[-1]
        energies = numpy.arange(core_loss_range_eV[0],core_loss_range_eV[1],delta)
        print(edge_data[4].size,edge_data[3].size,edge_data[1].size)
        plt.plot(edge_data[4],edge_data[1][0])
        plt.plot(edge_data[4],edge_data[3][0])
        plt.plot(energies,core_loss_spectra)
        plt.show()

    # The following should ultimately be pulled out of the edge ID table, based on atomic number and edge onset
    shell_number = 1
    subshell_index = 1
    cross_section_data,diff_cross_section,egrid_ev = eels_analysis.partial_cross_section_nm2(atomic_number, shell_number, subshell_index, edge_onset_eV, edge_delta_eV,
                                                                    beam_energy_eV, convergence_angle_rad, collection_angle_rad)
    #print("cross_section_data", cross_section_data, edge_data[0])
    cross_section = cross_section_data #cross_section_data[0]
    atomic_abundance = numpy.where(cross_section > 0.0, edge_data[0] / cross_section, numpy.zeros_like(edge_data[0]))

    # Now find errors. Assume Poisson error for experimental cross section (S_exp), 10% theoretical errors (s_thy). Then error in relative atomic abundance is
    # S = atomic_abundance*\sqrt[ (S_exp/cross_exp)^2 + (S_thy/cross_thy)^2 ]
    # Poisson error is sqrt of integral of total experimental signal. Total integral is = edge_data[1]
    sig_sq_exp = edge_data[2]
    relative_error_thy = 0.1 # 10% error
    print(edge_data[0])
    i=0
    sig_sq_over_cross = numpy.where(edge_data[0] > 0.0, sig_sq_exp/edge_data[0]**2, 0.0)
    error = atomic_abundance*numpy.sqrt(sig_sq_over_cross + relative_error_thy**2)
    
    return atomic_abundance, error, edge_data, numpy.array(diff_cross_section)*atomic_abundance[0], egrid_ev

def stoichiometry_from_eels(eels_spectrum: numpy.ndarray, energy_range_ev: numpy.ndarray, background_ranges_ev: typing.List[numpy.ndarray], atomic_numbers: typing.List[int],
                                 edge_onsets_ev: typing.List[typing.List[float]], edge_deltas_ev: typing.List[float], 
                                 beam_energy_ev: float, convergence_angle_rad: float, collection_angle_rad: float):
    """Quantify a complete EELS spectrum given atomic species in the system and edges in the spectrum (signal ranges and background ranges).
    Input: 
        eels_spectrum: numpy array of specral data. For this to be sensible, the array should be over a range that includes more than one edge.
        energy_range_ev  - 2 element array containing energy range of the spectrum. First element is energy associated with start of
                           first spectral element, second element is end of last spectral element, i.e. = # of elements * energy_step + energy_range_ev[0]
        background_ranges_ev - list of numpy arrays, each one specifying the energy range for background subtraction. There should be one range for each element of edge_onsets_ev.
        atomic_numbers: atomic number associated with each edge onset in edge_onsets_ev.
        edge_onsets_ev: experimental edge energies associated with experimental spectrum
        edge_deltas_ev: signal width for each edge
        beam_energy_ev: electron beam energy in ev
        convergence_angle_rad: convergence angle in radians
        collection_angle_rad: collection angle in radians

    Returns:
        (stoichiometry,error_in_stoichiometry, edge_data, diff_cross_section, egrid_ev)

              stoichiometry - relative to first atom in list.
              error_in_stoichiometry - combined theoretical/experimental errors.
              edge_data - list of tuples: for each edge listed in stoichiometry, (signal_integral, signal_profile, total_integral, background_model, energy_grid)
                                          signal_integral  - core-loss edge signal integral
                                          signal_profile   - background subtracted core-loss profile
                                          total_integral   - total integral (over the signal range) before background subtraction.
                                          background_model - background function evaluated on same grid as signal_profile.
                                          energy_grid      - grid on which the background model and signal profile are defined.
              diff_cross_section - theoretical differential cross section.
              egrid_ev - energy grid on which the diff_cross_section is calculated
    """
    # For now assert that the number of atomic species, background_ranges, edge_onsets, edge_deltas are equal. This assumes that each
    # edge range only contains signal from one atomic species.
    assert len(edge_onsets_ev) >=1
    assert len(atomic_numbers) >=1
    assert len(edge_onsets_ev) == len(edge_deltas_ev)
    assert len(edge_onsets_ev) == len(background_ranges_ev)


    # First calculate the cross section associated with each edge. 
    # Loop over atoms in the spectrum.
    iAtom = 0
    if eels_spectrum.ndim == 1:
        image_shape = 1
    else:
        image_shape = eels_spectrum.shape[0]

    
    abundance = numpy.array([[0.0]*image_shape]*len(atomic_numbers))
    edge_data = []
    diff_cross_section = []
    egrid_ev = []
    err = numpy.array([[0.0]*image_shape]*len(atomic_numbers))
    stoichiometry = numpy.array([[0.0]*image_shape]*len(atomic_numbers))
    total_number  = numpy.array([0.0]*image_shape)
    total_error  = numpy.array([0.0]*image_shape)
    error_in_stoichiometry=numpy.array([[0.0]*image_shape]*len(atomic_numbers))
    for atomic_number in atomic_numbers:
        # Find relative atomic abundance for this edge.
        #print(atomic_number, edge_onsets_ev[iAtom])
        abundance[iAtom],err[iAtom],ed,diffx,egrid = relative_atomic_abundance(eels_spectrum, energy_range_ev, background_ranges_ev[iAtom][:],
                                                         atomic_number, edge_onsets_ev[iAtom], edge_deltas_ev[iAtom], beam_energy_ev,
                                                         convergence_angle_rad, collection_angle_rad)

        edge_data = edge_data + [ed]
        diff_cross_section = diff_cross_section + [diffx]
        egrid_ev = egrid_ev + [egrid]
        stoichiometry[iAtom] = abundance[iAtom]
        total_number = total_number + abundance[iAtom]
        total_error  = numpy.sqrt(total_error**2 + err[iAtom]**2)
        iAtom += 1

    iAtom = 0
    for atomic_number in atomic_numbers:
        stoichiometry[iAtom] = numpy.where(total_number > 0.0, abundance[iAtom]/total_number, -1)

        # sum relative errors in quadrature.
        error_in_stoichiometry[iAtom] = numpy.where(abundance[iAtom] > 0, stoichiometry[iAtom]*numpy.sqrt((err[iAtom]/abundance[iAtom])**2 + (total_error/total_number)**2), 0.0)
            
        iAtom += 1

    return stoichiometry,error_in_stoichiometry, edge_data, diff_cross_section, egrid_ev
            
            
        
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
