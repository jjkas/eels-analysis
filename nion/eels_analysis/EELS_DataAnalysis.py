"""
    EELS Data Analysis

    A library of functions for EELS data analysis.
"""

# third party libraries
import numpy
import scipy

# local libraries
from . import CurveFittingAndAnalysis
from . import EELS_CrossSections

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
    # Sanity checks:
    assert energy_range_ev[0] < energy_range_ev[1]

    if search_range_ev is not None:
        assert search_range_ev[0] < search_range_ev[1]

    # Calulate energy step
    energy_step = (energy_range_ev[1] - energy_range_ev[0])/eels_spectrum.shape[-1]

    # Set minimum value of search range. Use 60eV if nothing is passed by user, i.e., cut out low loss.
    emin_search = max((search_range_ev[0],energy_range_ev[0])) if search_range_ev is not None else max((60.0,energy_range_ev[0]))

    # Set maximum value of search range. Use max of spectrum if noting is passed by user.
    emax_search = min((search_range_ev[1],energy_range_ev[1])) if search_range_ev is not None else energy_range_ev[1]

    # Find closest indices corresponding to emin_search and emax_search
    imin_search = int((emin_search - energy_range_ev[0])/energy_step)
    imax_search = int((emax_search - energy_range_ev[0])/energy_step)

    # Create cut energy and spectral arrays.
    eels_spectrum_search = eels_spectrum[imin_search:imax_search]
    energies_search      = numpy.arange(emin_search,emax_search)

    assert eels_spectrum_search.shape[-1] == energies_search.shape[-1]

    # Here we will search for steplike features by creating an array of correlation coefficients correlating the energy axis
    # with the spectrum. Background signal will generally be negatively correlated with energy, while strong edges will be
    # positively correlated with energy, and weak edges will just show an increase in correlation.
    
    # Create space for correlation array
    corr_coeff_array = numpy.zeros(eels_spectrum_search.shape[-1])

    # Set sensitivity parameter - used for general control of sensitivity of the algorithm to step like features.
    # Lower sensitivity will cause the algorithm to find fewer edges, while larger sensitivity will cause it to
    # find more edges. Ensure sensitivity is between 0 and 1.
    sensitivity_parameter = 1.0 - max((min((kwargs.get('sensitivity',0.5),0.0)),0.0))
    
    # Set energy range for calculating the correlation coefficient. A smaller energy range will be more
    # susceptible to noise, while a larger energy range will only be sensitive to changes the persist over
    # a larger energy range. Default to 15 eV
    correlation_energy_range = kwargs.get("correlation_energy_range_ev",30.0*sensitivity_parameter)

    for i_energy, energy in enumerate(energies_search):
        # Create correlation coefficient between subarrays corresponding to current energy point +/- correlation_energy_range
        # coefficient is between -1 and 1, but we'll add 1 and devide by 2 to get a parameter that ranges from 0 to 1.
        corr_coeff_array[i] = (numpy.corrcoef(energies_search[numpy.where(abs(energies_search < energies_search[i_energy]) < correlation_energy_range)],
                                          eels_spectrum_search[numpy.where(abs(energies_search < energies_search[i_energy]) < correlation_energy_range)]) + 1.0)/2.0

    # Take second derivative of correlation coefficients - local maxima in second derivatives are usually slightly below the edge.
    corr_coeff_d2 = numpy.gradient(numpy.gradient(corr_coeff_array,energies_search))
                            
    # We will take only first edge found after each maximum in the corr_coeff_array - energy_range_unique.
    # Thus energy_range_unique specifies the separation required between edges. A small energy_range_unique
    # will be sensitive to fine-structure, while a large energy_range_unique will miss overlapping edges.
    # Default is 50eV.
    edge_separation_energy = kwargs.get('edge_separation_energy',100.0*sensitivity_parameter)

    # Define the cutoff correlation coefficient below which we will ignore all maxima.
    # Default is 0.6*maximum correlation found.
    correlation_cutoff_scale = kwargs.get('correlation_cutoff_scale',0.6*sensitivity_parameter*2.0) 
    minCorr = numpy.amax(corr_coeff_array)*correlation_cutoff_scale

    # Find the indices of maxima in the correlation coefficient which are greater than minCorr
    ind_max_corr = scipy.signal.argrelextrema(corr_coeff_array[numpy.where(corr_coeff_array > minCorr)], numpy.greater)

    # find max in second derivative of corr_coeff_array between each max in corr_coeff_array edge_separation_energy below it.
    i=0
    edge_energies = numpy.zeros(ind_max_corr.size)
    number_of_edges = 1
    for energy_of_max in numpy.nditer(energies_search[numpy.where(corr_coeff_array > minCorr)][ind_max_corr]):
        ecut = energies_search[numpy.where(
            numpy.logical_and(energies_search > energy_of_max - edge_separation_energy, energies_search < energy_of_max))]
        d2cut = corr_coeff_d2[numpy.where(
            numpy.logical_and(energies_search > energy_of_max - edge_separation_energy, energies_search < energy_of_max))]
        edge_energies[nEdge-1] = ecut[numpy.where(d2cut == numpy.amax(d2cut))]
        nEdge+=1

    # Return only unique edges as the algorithm can find the same edge more than once.
    return numpy.unique(edge_energies[:nEdge-1])
                                        
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
