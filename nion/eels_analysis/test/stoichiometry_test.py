import numpy
import unittest
from pathlib import Path
#from EELS_DataAnalysis import stoichiometry_from_eels
from nion.eels_analysis import EELS_DataAnalysis 
from atomic_eels import atomic_diff_cross_section
#import matplotlib.plotly as plt

class TestLibrary(unittest.TestCase):

    def test_stoichiometry_found_from_theoretical_EELS(self):
        # Make BN eels data using FEFF.
        atomic_numbers=[5,7]
        amps =[1.0,1.0]
        edge_label = 'K'
        beam_energy_keV = 200.0
        convergence_angle_mrad = 1.5
        collection_angle_mrad = 1.5
        egrid_eV = numpy.arange(0.0, 1000.0, 0.5) # Define energy grid
        energyDiffSigma_total = numpy.zeros_like(egrid_eV) # Initialize total cross section.
        edge_onsets = [0.0, 0.0]
        edge_deltas = [0.0, 0.0]
        background_ranges = [numpy.zeros(2),numpy.zeros(2)]
        iEdge = 0
        for atomic_number in atomic_numbers:
            #print(atomic_number,edge_label,beam_energy_keV,convergence_angle_mrad,collection_angle_mrad)
            energyDiffSigma,edge_onsets[iEdge] = atomic_diff_cross_section(atomic_number, edge_label, beam_energy_keV,
                                                                           convergence_angle_mrad, collection_angle_mrad, egrid_eV)
            energyDiffSigma_total = numpy.add(energyDiffSigma_total,energyDiffSigma*amps[iEdge])
            # set background ranges and offsets, etc.
            background_ranges[iEdge][0] = max(edge_onsets[iEdge]-30.0,0.0)
            background_ranges[iEdge][1] = max(edge_onsets[iEdge]-5.0,0.0)
            edge_deltas[iEdge] = 30.0
            iEdge += 1
        
        #print('bgr',background_ranges,edge_onsets)
        bgfunc = lambda x: 1.0e-3/(x+10.0)**3
        background = numpy.vectorize(bgfunc)
        energyDiffSigma_total = numpy.add(energyDiffSigma_total,background(egrid_eV))*10.0e12
        #plt.plot(egrid_eV,energyDiffSigma_total)
        #plt.show()
        #noise = numpy.random.normal(0.0, max(energyDiffSigma_total)/100.0,energyDiffSigma_total.size)
        #energyDiffSigma_total = numpy.add(noise,energyDiffSigma_total)
        
        erange = numpy.zeros(2)
        erange[0] = egrid_eV[0]
        erange[1] = egrid_eV[-1]
        
        stoichiometry = EELS_DataAnalysis.stoichiometry_from_eels(energyDiffSigma_total,erange,background_ranges,atomic_numbers,edge_onsets,edge_deltas,
                                                beam_energy_keV*1000.0, convergence_angle_mrad/1000.0, collection_angle_mrad/1000.0)
        
        iAtom = 0
        print("Stoichiometry from theoretical EELS signal of BN:")
        for atomic_number in atomic_numbers:
            print(atomic_number, stoichiometry[iAtom])            
            assert abs(stoichiometry[iAtom]/amps[iAtom]*amps[0]-1.0) < 0.01 # Test stoichiometry found is better than 1%
            iAtom += 1    


    def test_stoichiometry_found_from_experimental_eels(self):
        # Read EELS data from file (BN). This is data from Tracy, taken from a thin part of the sample,
        # and represents to some extent a best case scenario.
        data_file = Path('./Test_Data/eelsdbBN.csv')
        #data_file = Path('./Test_Data/EELS_Thick.csv')
        #data_file = Path('./Test_Data/EELS_Thin.csv')
        
        energy_grid,spectrum = numpy.loadtxt(data_file, delimiter=',',unpack=True)
        
        # Set up input to stoichiometry quantification. All settings are hard coded to
        # BN to match DM 2.32.888.
        atomic_numbers=[5,7]
        edge_label = 'K'
        beam_energy_keV = 200.0
        convergence_angle_mrad = 0.0
        collection_angle_mrad = 100.0     
        edge_onsets = [188.0, 401.0]
        edge_deltas = [25.0, 25.0]
        background_ranges = [numpy.zeros(2),numpy.zeros(2)]
        background_ranges[0][0] = 167.0
        background_ranges[0][1] = 183.0
        background_ranges[1][0] = 358.0
        background_ranges[1][1] = 393.0
            
        erange = numpy.zeros(2)
        erange[0] = energy_grid[0]
        erange[1] = energy_grid[-1]
        
        stoichiometry = EELS_DataAnalysis.stoichiometry_from_eels(spectrum,erange,background_ranges,atomic_numbers,edge_onsets,edge_deltas,
                                                beam_energy_keV*1000.0, convergence_angle_mrad/1000.0, collection_angle_mrad/1000.0)
        
        iAtom = 0
        print('Stoichiometry for BN from experimental EELS data from the EELS Atlas:')
        for atomic_number in atomic_numbers:
            print(atomic_number, stoichiometry[iAtom])            
            #assert abs(stoichiometry[iAtom]/amps[iAtom]*amps[0]-1.0) < 0.01 # Test stoichiometry found is better than 1%
            iAtom += 1

            
if __name__ == '__main__':
    unittest.main()

