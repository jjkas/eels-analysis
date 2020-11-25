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
# import matplotlib.pyplot as plt

# niondata must be available as a module.
# it can be added using something similar to
#   conda dev /path/to/niondata
from nion.data import Calibration
from nion.data import DataAndMetadata

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

# Note: EELSAnalysis is only available in sys.path above is appended with its _parent_ directory.
from nion.eels_analysis import PeriodicTable
from nion.eels_analysis import EELS_DataAnalysis


class TestEELSAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass


    def test_find_species_from_experimental_edge_data(self):
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
            elements_exp = [elem.strip(string.digits) for elem in re.findall('[A-Z][^A-Z]*',chem_formula) if str(ptable.atomic_number(elem)) in ptable.find_elements_in_energy_interval(search_range)]
            print(file_name,':')
            experimental_edge_data = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev,search_range, debug_plotting = False)
            df=pandas.read_csv(edge_file, delim_whitespace=True, header=None)

            edge_data = EELS_DataAnalysis.find_species_from_experimental_edge_data(eels_spectrum, energy_range_ev, experimental_edge_data, search_range_ev = search_range, only_major_edges = True) 
            elements_found = [ed[0] for ed in edge_data]
            edge_data = EELS_DataAnalysis.find_species_from_experimental_edge_data(eels_spectrum, energy_range_ev, experimental_edge_data, search_range_ev = search_range, only_major_edges = False, element_list = elements_found) 
            edge_energies = experimental_edge_data[0]
            # Print edges found, and edges found by visual inspection for comparison.
            # The edge finder will not find all edges necessarily, and might find extra edges.
            print("Number of edges found = ", edge_energies.size)
            print("Edges found:")
            print(numpy.sort(edge_energies).astype(int))
            elements_found = [ptable.element_symbol(ed[0]) for ed in edge_data ]
            ntot += 1
            if(all(x in elements_found for x in elements_exp)):
                print(chem_formula, ":  PASS")
            else:
                print(chem_formula, ":  FAIL")
                print("Missing elements:")
                for x in elements_exp:
                    if x not in elements_found:
                        print(x)
                nfail += 1
            for ed in sorted(edge_data, key=lambda x: x[0]):
                print(PeriodicTable.PeriodicTable().element_symbol(ed[0]),ed)

            i_search += 1
        print('Percentage failure:', float(nfail)/float(ntot)*100.0)

if __name__ == '__main__':
    unittest.main()
