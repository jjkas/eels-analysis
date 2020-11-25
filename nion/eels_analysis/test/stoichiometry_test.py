import numpy
import unittest
from pathlib import Path
#from EELS_DataAnalysis import stoichiometry_from_eels
from nion.eels_analysis import EELS_DataAnalysis 
from nionswift_plugin.feff_interface import FEFF_EELS_Service

class TestLibrary(unittest.TestCase):

    def test_stoichiometry_found_from_theoretical_EELS(self):
        if True:
            return

        from atomic_eels import atomic_diff_cross_section
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
        estep = (egrid_eV[-1] - egrid_eV[0])/(egrid_eV.size - 1)
        print(estep)
        erange[0] = egrid_eV[0]
        erange[1] = egrid_eV[-1] + estep
        
        stoich_data = EELS_DataAnalysis.stoichiometry_from_eels(energyDiffSigma_total,erange,background_ranges,atomic_numbers,edge_onsets,edge_deltas,
                beam_energy_keV*1000.0, convergence_angle_mrad/1000.0, collection_angle_mrad/1000.0)
        stoichiometry = stoich_data[0]
        error_in_stoichiometry = stoich_data[1]
        
        
        iAtom = 0
        print("Stoichiometry from theoretical EELS signal of BN:")
        for atomic_number in atomic_numbers:
            print(atomic_number, stoichiometry[iAtom][0], '+/-', error_in_stoichiometry[iAtom][0])            
            assert abs(stoichiometry[iAtom][0]/amps[iAtom]*amps[0]-0.5) < 0.01 # Test stoichiometry found is better than 1%
            iAtom += 1    

    def test_stoichiometry_from_multidimensional_EELS(self):
        # Make BN eels data using FEFF.
        atomic_numbers=[5,7]
        nSpectra = 100
        amps =[1.0,1.0]
        edge_label = 'K'
        beam_energy_keV = 200.0
        convergence_angle_mrad = 1.5
        collection_angle_mrad = 1.5
        egrid_eV = numpy.arange(0.0, 1000.0, 0.5) # Define energy grid
        energyDiffSigma_total = numpy.array([numpy.zeros_like(egrid_eV)]*nSpectra) # Initialize total cross section.
        energyDiffSigma = numpy.array([numpy.zeros_like(egrid_eV)]*2) # Initialize total cross section.
        edge_onsets = [0.0, 0.0]
        edge_deltas = [0.0, 0.0]
        background_ranges = [numpy.zeros(2),numpy.zeros(2)]
        iEdge = 0
        for atomic_number in atomic_numbers:
            energyDiffSigma[iEdge],edge_onsets[iEdge] = atomic_diff_cross_section(atomic_number, edge_label, beam_energy_keV,
                                                                           convergence_angle_mrad, collection_angle_mrad, egrid_eV)
            iEdge += 1
            
        iEdge = 0
        for atomic_number in atomic_numbers:
            iSpectrum = 0
            while iSpectrum < nSpectra:

                if iEdge == 0: 
                    amps[iEdge] = numpy.sin(float(iSpectrum)/float(nSpectra)*numpy.pi/2.0)**2
                else:
                    amps[iEdge] = numpy.cos(float(iSpectrum)/float(nSpectra)*numpy.pi/2.0)**2
                
                energyDiffSigma_total[iSpectrum] = numpy.add(energyDiffSigma_total[iSpectrum],energyDiffSigma[iEdge]*amps[iEdge])
                iSpectrum += 1
                
            # set background ranges and offsets, etc.
            background_ranges[iEdge][0] = max(edge_onsets[iEdge]-30.0,0.0)
            background_ranges[iEdge][1] = max(edge_onsets[iEdge]-5.0,0.0)
                
            edge_deltas[iEdge] = 30.0
            iEdge += 1

        iSpectrum = 0
        # Add background.
        bgfunc = lambda x: 1.0e-3/(x+10.0)**3
        background = numpy.vectorize(bgfunc)

        while iSpectrum < nSpectra:
            energyDiffSigma_total[iSpectrum] = numpy.add(energyDiffSigma_total[iSpectrum],background(egrid_eV))*10.0e12
            iSpectrum += 1
            
        #print('bgr',background_ranges,edge_onsets)

        #plt.plot(egrid_eV,energyDiffSigma_total)
        #plt.show()
        #noise = numpy.random.normal(0.0, max(energyDiffSigma_total)/100.0,energyDiffSigma_total.size)
        #energyDiffSigma_total = numpy.add(noise,energyDiffSigma_total)
        
        erange = numpy.zeros(2)
        erange[0] = egrid_eV[0]
        erange[1] = egrid_eV[-1]
        
        stoich_data = EELS_DataAnalysis.stoichiometry_from_eels(energyDiffSigma_total,erange,background_ranges,atomic_numbers,edge_onsets,edge_deltas,
                beam_energy_keV*1000.0, convergence_angle_mrad/1000.0, collection_angle_mrad/1000.0)
        stoichiometry = stoich_data[0]
        error_in_stoich = stoich_data[1]
        
        
        iAtom = 0
        print("Stoichiometry from multidimensional spectrum array.")
        for atomic_number in atomic_numbers:
            print(atomic_number)
            print(stoichiometry[iAtom])

            iAtom += 1

        #xvals=numpy.arange(0,100,1)
        #plt.plot(xvals,stoichiometry[0]-numpy.sin(xvals/100.0*numpy.pi/2.0)**2,xvals,stoichiometry[1]-numpy.cos(xvals/100.0*numpy.pi/2.0)**2)
        #plt.show()

            
    def test_stoichiometry_found_from_experimental_eels(self):
        # Read EELS data from file (BN). This is data from Tracy, taken from a thin part of the sample,
        # and represents to some extent a best case scenario.
        test_data_dir = Path(__file__).parent / 'Test_Data'
        data_files = [test_data_dir / 'BN_0-0910eV_.msa', test_data_dir / 'CaCO3.msa', test_data_dir / 'CuO.msa']
        labels = ['BN', 'CaCO_3','CuO']
        #data_file = Path('./Test_Data/EELS_Thick.csv')
        #data_file = Path('./Test_Data/EELS_Thin.csv')
        

        atomic_number_arrays = [[7,5],[8,20,6],[29,8]]
        beam_energies        = [200.0,200.0,200.0]
        convergence_angles   = [0.0, 0.0, 0.0]
        collection_angles    = [100.0, 100.0, 100.0]
        edge_onset_arrays    = [[401.0, 188.0],[532.0, 346.0, 284.0], [931.0,532.0]]
        edge_delta_arrays    = [[25.0,25.0],[40.0,25.0,25.0],[40.0,40.0]]
        background_arrays    = [ [numpy.array([358.0,393.0]),numpy.array([167.0,183.0])], [numpy.array([474.0, 521.0]),numpy.array([308.0,339.0]),numpy.array([253.0,278.0])],
                                 [numpy.array([831.0,912.0]),numpy.array([474.0,521.0])]]
        DM_Stoichiometries = [[1.0, 0.83], [1.0, 0.76, 0.27],[1.0,0.09]]
        True_Stoichiometries = [[1.0,1.0],[1.0, 0.3333, 0.3333],[1.0,1.0]]

        iData = 0
        for data_file in data_files:
            energy_grid,spectrum = numpy.loadtxt(data_file, delimiter=',',unpack=True)
            # Set up input to stoichiometry quantification. All settings are hard coded to
            # BN to match DM 2.32.888.
            beam_energy_keV = beam_energies[iData]
            atomic_numbers  = atomic_number_arrays[iData]
            convergence_angle_mrad = convergence_angles[iData]
            collection_angle_mrad = collection_angles[iData]
            edge_onsets = edge_onset_arrays[iData]
            edge_deltas = edge_delta_arrays[iData]
            background_ranges = background_arrays[iData]
            
            erange = numpy.zeros(2)
            erange[0] = energy_grid[0]
            erange[1] = energy_grid[-1]

            stoich_data = EELS_DataAnalysis.stoichiometry_from_eels(spectrum,erange,background_ranges,atomic_numbers,edge_onsets,edge_deltas,
                    beam_energy_keV*1000.0, convergence_angle_mrad/1000.0, collection_angle_mrad/1000.0)
            stoich = stoich_data[0]
            error_in_stoich = stoich_data[1]
        
            iAtom = 0
            DM_Stoichometry = DM_Stoichiometries[iData]
            True_Stoichiometry = True_Stoichiometries[iData]
            print("----------------------------------------------------------------------------\n\n\n")
            print('Stoichiometry from experimental EELS data from the EELS Atlas:' + labels[iData])
            print('atomic#, N, N from DM, True N')
            for atomic_number in atomic_numbers:
                print(atomic_number, stoich[iAtom][0],'+/-',error_in_stoich[iAtom][0],DM_Stoichometry[iAtom], True_Stoichiometry[iAtom])

                iAtom += 1

            print("----------------------------------------------------------------------------")
            iData += 1
            

if __name__ == '__main__':
    unittest.main()

