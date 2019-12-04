import numpy
import unittest
#from EELS_DataAnalysis import stoichiometry_from_eels
from nion.eels_analysis import EELS_DataAnalysis 
from atomic_eels import atomic_diff_cross_section
#import matplotlib.plotly as plt

class TestLibrary(unittest.TestCase):

    def test_stoichiometry(self):
        # Make BN eels data using FEFF.
        atomic_numbers=[5,7]
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
            energyDiffSigma_total = numpy.add(energyDiffSigma_total,energyDiffSigma)
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
        for atomic_number in atomic_numbers:
            #print(atomic_number, stoichiometry[iAtom])
            assert abs(stoichiometry[iAtom]-1.0) < 0.01
            iAtom += 1    

if __name__ == '__main__':
    unittest.main()

