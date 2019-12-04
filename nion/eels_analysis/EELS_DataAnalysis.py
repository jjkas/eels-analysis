"""
    EELS Data Analysis

    A library of functions for EELS data analysis.
"""

# third party libraries
import numpy
import typing

# local libraries
from nion.eels_analysis import CurveFittingAndAnalysis
from nion.eels_analysis import EELS_CrossSections
from nion.eels_analysis import PeriodicTable

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

def stoichiometry_from_eels(eels_spectrum: numpy.ndarray, eels_range_eV: numpy.ndarray, background_ranges_eV: typing.List[numpy.ndarray], atomic_numbers: typing.List[int],
                                 edge_onsets_eV: typing.List[float], edge_deltas_eV: typing.List[float], 
                                 beam_energy_eV: float, convergence_angle_rad: float, collection_angle_rad: float) -> typing.List[float]:
    """Quantify a complete EELS spectrum given atomic species in the system and edges in the spectrum (signal ranges and background ranges).

    Returns:
        stoichiometries - relative to first atom in list.
        calculated cross sections - in nm^2.
    """
    # For now assert that the number of atomic species, background_ranges, edge_onsets, edge_deltas are equal. This assumes that each
    # edge range only contains signal from one atomic species.
    assert len(edge_onsets_eV) > 1
    assert len(edge_onsets_eV) == len(atomic_numbers)
    assert len(edge_onsets_eV) == len(edge_deltas_eV)
    assert len(edge_onsets_eV) == len(background_ranges_eV)


    # First calculate the cross section associated with each edge. 
    # Loop over atoms in the spectrum.
    iAtom = 0
    abundance=[0.0]*len(atomic_numbers)
    stoichiometry=[0.0]*len(atomic_numbers)
    
    for atomic_number in atomic_numbers:
        # Find relative atomic abundance for this edge.
        abundance[iAtom] = relative_atomic_abundance(eels_spectrum, eels_range_eV, background_ranges_eV[iAtom][:],
                                                         atomic_number, edge_onsets_eV[iAtom], edge_deltas_eV[iAtom], beam_energy_eV,
                                                         convergence_angle_rad, collection_angle_rad)
        
        stoichiometry[iAtom] = abundance[iAtom]/abundance[0]
        iAtom += 1

    return stoichiometry
            
            
        
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
