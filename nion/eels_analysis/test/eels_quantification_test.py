# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/Library_test.py

import numpy
import os
import random
import sys
import unittest
# The below are for more extensive testing of all eels atlas data.
import re
import string

# niondata must be available as a module.
# it can be added using something similar to
#   conda dev /path/to/niondata
from nion.data import Calibration
from nion.data import DataAndMetadata

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

# Note: EELSAnalysis is only available in sys.path above is appended with its _parent_ directory.
from nion.eels_analysis import PeriodicTable
from nion.eels_analysis import EELS_DataAnalysis
from nionswift_plugin.feff_interface import FEFF_EELS_Service


class TestEELSAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass


    def test_eels_quantification(self):
        if True: # For now turn this test off. I will keep the directory of all EELS Atlas data on hand for more testing.
            return

        import glob
        import pandas 
        df=pandas.read_csv("EELS_Atlas_Major/files_HE.dat", delim_whitespace=True, header=None)
        file_names,tmp1,efin,tmp2,estart=df.to_numpy().T

        i_search=0
        total_extra_found = 0
        total_edges_edb = 0
        total_edges_matched = 0
        mintol=0.1
        nfail = 0
        ntot = 0
        ptable = PeriodicTable.PeriodicTable()
        for file_name in file_names:
           # Load data from text file.
            #data_file = glob.glob('./EELS_Atlas_Major/**/' + file_name,recursive = True)[0]
            print('\n\n\n')
            data_file = glob.glob('./EELS_Atlas_Major/**/' + file_name,recursive = True)[0]

            #edge_file = glob.glob('./EELS_Atlas_Major/**/' + 'edges_' + os.path.splitext(file_name)[0]+'.dat', recursive=True)[0]
            edge_file = glob.glob('./EELS_Atlas_Major/**/' + 'edges_' + os.path.splitext(file_name)[0]+'.dat', recursive=True)[0]
            energies, eels_spectrum = numpy.loadtxt(data_file, delimiter=',',unpack=True)
            energy_step = (energies[-1] - energies[0])/energies.size
            energy_range_ev = numpy.array([energies[0],energies[-1]+energy_step])



            if estart[i_search] < 0:
                estart[i_search] = tmp1[i_search]

            search_range = [estart[i_search],efin[i_search]]
            chem_formula = file_name.split('_')[0]
            elements_exp = [elem.strip(string.digits) for elem in re.findall('[A-Z][^A-Z]*',chem_formula) if str(ptable.atomic_number(elem.strip(string.digits))) in ptable.find_elements_in_energy_interval(search_range)]
            print(file_name,':')
            
            experimental_edge_data = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev,search_range, debug_plotting = False)
            df=pandas.read_csv(edge_file, delim_whitespace=True, header=None)

            edge_data = EELS_DataAnalysis.find_species_from_experimental_edge_data(eels_spectrum, energy_range_ev, experimental_edge_data, search_range_ev = search_range, only_major_edges = True) 
            elements_found = [ed[0] for ed in edge_data]
            edge_data = EELS_DataAnalysis.find_species_from_experimental_edge_data(eels_spectrum, energy_range_ev, experimental_edge_data, search_range_ev = search_range, only_major_edges = False, element_list = elements_found) 
            edge_energies = experimental_edge_data[0]
            # Print edges found, and edges found by visual inspection for comparison.
            # The edge finder will not find all edges necessarily, and might find extra edges.
            elements_found = [ptable.element_symbol(ed[0]) for ed in edge_data ]
            ntot += 1
            if(all(x in elements_found for x in elements_exp)):
                missing=False
            else:
                print(chem_formula, ":  Failed to find all elements")
                i_search += 1
                continue

            edge_data = [ed for ed in edge_data if ptable.element_symbol(ed[0]) in elements_exp]
            #print('edge_data', edge_data)

            i_search += 1

            
            # We now have the elements and the edges we want to analyze, lets do the quantification
            # Set microscope parameters
            beam_energy_keV = 200.0
            convergence_angle_mrad = 0.0
            collection_angle_mrad = 100.0

            # Set up the atomic numbers, edge onsets, and background ranges
            atomic_numbers = []
            background_ranges = []
            edge_onsets = []
            edge_deltas = []
            iElement = 0
            for ed in edge_data:
                iEdge = 0
                edge_onset = []
                while iEdge < len(ed[1]):
                    edge_onsets = edge_onsets + [ed[1][iEdge][2]]
                    atomic_numbers = atomic_numbers + [ed[0]]
                    iEdge += 1

            #print('Atoms in system:', atomic_numbers)
            deltas = 30.0
            for i,onset in enumerate(edge_onsets):
                if i+1 < len(edge_onsets) and atomic_numbers[i] == atomic_numbers[i+1]:
                    if edge_onsets[i+1] - onset > deltas:
                        edge_deltas = edge_deltas + [deltas]
                    else:
                        edge_deltas = edge_deltas + [deltas]#[edge_onsets[i+1] - onset]                        
                        #print(edge_deltas[-1])
                else:
                    edge_deltas = edge_deltas + [min(deltas,energy_range_ev[1]-onset)]

                if i > 0:
                    if atomic_numbers[i] == atomic_numbers[i-1]:
                        #background_ranges = background_ranges + [numpy.array([max(onset - 30.0,edge_onsets[i-1]), onset - 10.0])]
                        background_ranges = background_ranges + [numpy.array([max(onset - 30.0,energy_range_ev[0]), onset - 10.0])]
                    else:
                        background_ranges = background_ranges + [numpy.array([max(onset - 30.0,energy_range_ev[0]), onset - 10.0])]
                else:
                    background_ranges = background_ranges + [numpy.array([max(onset - 30.0,energy_range_ev[0]), onset - 10.0])]

            #print(edge_onsets)
            #print(edge_deltas)
            #print(background_ranges)
            stoich,error_in_stoich,quant_data,diff_cross,egrid_ev = EELS_DataAnalysis.stoichiometry_from_eels(eels_spectrum,energy_range_ev,background_ranges,atomic_numbers,edge_onsets,edge_deltas,
                                                                      beam_energy_keV*1000.0, convergence_angle_mrad/1000.0, collection_angle_mrad/1000.0)
            for iat,atm in enumerate(atomic_numbers):
                erange=(edge_onsets[iat]-50.0,edge_onsets[iat] + edge_deltas[iat] + 50)
                edges=ptable.find_all_edges_in_energy_interval(erange,atm)
                print(ptable.element_symbol(atm),':')
                print('Energy Range: ', edge_onsets[iat],edge_onsets[iat]+edge_deltas[iat])
                for edg in edges:
                    edgestr=edg.get_shell_str_in_eels_notation(include_subshell=True) 
                    print('\t',edgestr)
                print('\t',stoich[iat],error_in_stoich[iat])
                print('\n\n###############################################')

                if False:
                    import matplotlib.pyplot as plt
                    e_step = (quant_data[iat][4][1] - quant_data[iat][4][0])/quant_data[iat][1][0].size
                    profile_grid = numpy.arange(quant_data[iat][4][0],quant_data[iat][4][1],e_step)
                    plt.plot(profile_grid,quant_data[iat][1][0])
                    plt.plot(profile_grid,quant_data[iat][3][0])
                    plt.plot(energies,eels_spectrum)
                    plt.xlim(quant_data[iat][4][0]-50,quant_data[iat][4][1]+50)
                    plt.plot(egrid_ev[iat],diff_cross[iat])
                    plt.show()
        

if __name__ == '__main__':
    unittest.main()
