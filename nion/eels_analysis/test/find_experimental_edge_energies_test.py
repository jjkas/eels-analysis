# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/Library_test.py

import numpy
import os
import random
import sys
import unittest
# The below are for more extensive testing of all eels atlas data.
import glob
import pandas 
import matplotlib.pyplot as plt

# niondata must be available as a module.
# it can be added using something similar to
#   conda dev /path/to/niondata
from nion.data import Calibration
from nion.data import DataAndMetadata

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

# Note: EELSAnalysis is only available in sys.path above is appended with its _parent_ directory.
from nion.eels_analysis import EELS_DataAnalysis


class TestEELSAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass


    def test_find_experimental_edge_energies_with_defaults(self):
        # Tests the function without any optional parameters.s
        file_name = os.path.join('Test_Data','DectrisSI.csv')

        # Load data from file
        energies, eels_spectrum = numpy.loadtxt(file_name, delimiter=',',unpack=True)

        # Set energy range (goes from first energy to last energy + step
        energy_step = (energies[-1] - energies[0])/energies.size
        energy_range_ev = numpy.array([energies[0],energies[-1]+energy_step])

        # Find edges in the spectrum.
        edge_energies,q_factors = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev, debug_plotting=False)

        # Print edges found, and edges found by visual inspection for comparison. 
        # The edge finder will not find all edges necessarily, and might find extra edges.
        print("Number of edges found = ", edge_energies.size)
        print("Edges found:")
        print(numpy.sort(edge_energies).astype(int))
        print("Edges in spectrum")
        print(114,287,460,537,564,649,804,821,858,875)
        
    def test_find_experimental_edge_energies_EELS_Atlas(self):
        if True: # For now turn this test off. I will keep the directory of all EELS Atlas data on hand for more testing.
            return
        
        df=pandas.read_csv("EELS_Atlas_Major/files_HE.dat", delim_whitespace=True, header=None)
        file_names,tmp1,efin,tmp2,estart=df.to_numpy().T

        i_search=0
        total_extra_found = 0
        total_edges_edb = 0
        total_edges_matched = 0
        mintol=0.1
        for file_name in file_names:
           # Load data from text file.
            #data_file = glob.glob('./EELS_Atlas_Major/**/' + file_name,recursive = True)[0]
            data_file = glob.glob('./EELS_Atlas_Major/**/' + file_name,recursive = True)[0]
            
            #edge_file = glob.glob('./EELS_Atlas_Major/**/' + 'edges_' + os.path.splitext(file_name)[0]+'.dat', recursive=True)[0]
            edge_file = glob.glob('./EELS_Atlas_Major/**/' + 'edges_' + os.path.splitext(file_name)[0]+'.dat', recursive=True)[0]
            energies, eels_spectrum = numpy.loadtxt(data_file, delimiter=',',unpack=True)
            energy_step = (energies[-1] - energies[0])/energies.size
            energy_range_ev = numpy.array([energies[0],energies[-1]+energy_step])

            if estart[i_search] < 0:
                estart[i_search] = tmp1[i_search]

            search_range = [estart[i_search],efin[i_search]]
            print(file_name,':')
            edge_energies,q_factors = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev,search_range, debug_plotting = True)
            df=pandas.read_csv(edge_file, delim_whitespace=True, header=None)
            edge_names, edb_edge_energies = df.to_numpy().T
            edge_names = edge_names.tolist()
            new_q_factors = q_factors.tolist()
            match=[False]*edb_edge_energies.size
            matched=[False]*edge_energies.size
            i_edb = 0
            all_matched = True
            new_edge_energies = edge_energies[:].tolist()
            new_edb_edge_energies = edb_edge_energies[:].tolist()
            matched = []
            qfs=[]
            match_found = True
            print(new_edb_edge_energies)
            while len(new_edb_edge_energies) > 0 and len(new_edge_energies) > 0 and match_found:
            # Search trough all edb_edge_energy, edge_energy pairs to find minimum difference in energy
                min_diff = 1e10
                min_i = -1
                min_j = -1
                match_found = False
                for i, edb_edge_energy in enumerate(new_edb_edge_energies):
                    for j, energy in enumerate(new_edge_energies):
                        if min_diff > abs(energy - edb_edge_energy):
                            min_diff = abs(energy - edb_edge_energy)
                            min_i = i
                            min_j = j

                if min_diff/new_edb_edge_energies[min_i] < 0.03 or min_diff < 10.0:
                    # Set matched and remove elements from lists.
                    qfs = qfs + [new_q_factors.pop(min_j)]
                    matched = matched + [[edge_names.pop(min_i),new_edb_edge_energies.pop(min_i), new_edge_energies.pop(min_j)]]
                    match_found = True

            print("Edges matched:")
            print("Label  Atlas  Found")
            for m in matched:
                print(m)
            print("")
            print("Edges not matched:")
            i = 0
            while i <  len(new_edb_edge_energies):
                en = new_edb_edge_energies[i]
                if en > search_range[0] and en < search_range[1]:
                    print(edge_names[i], en)
                    i += 1
                else:
                    tmp = new_edb_edge_energies.pop(i)
                    tmp = edge_names.pop(i)
            print("")
            print("Extra edges found:")
            for en in new_edge_energies:
                print(en)
            total_edges_matched += len(matched)
            total_edges_edb += len(new_edb_edge_energies) + len(matched)
            total_extra_found += len(new_edge_energies)
            if len(new_edb_edge_energies) + len(matched) > 0:
                print("Percentage of edges matched:", float(len(matched))/float(len(matched)+len(new_edb_edge_energies))*100.0)
                print("Percentage extra edges:", float(len(new_edge_energies))/float(len(matched)+len(new_edb_edge_energies))*100.0)
                print("")
                print("")
                print("")
                if False:
                    ens = numpy.array([m[2] for m in matched])
                    qfs = numpy.array(qfs)
                    plt.stem(ens,qfs/numpy.amax(qfs),label='matched',use_line_collection=True)
                    plt.stem(numpy.array(edge_energies),numpy.array(q_factors)/numpy.amax(qfs),label='found',use_line_collection=True)
                    plt.plot(energies,eels_spectrum*energies**2/numpy.amax(eels_spectrum*energies**2))
                    plt.show()
            else:
                print("No edges in this energy range.")
                print("")
                print("")
                print("")

            i_search=i_search+1

        if total_edges_edb > 0:
            print("Total percentage of edges found: ", float(total_edges_matched)/float(total_edges_edb)*100.0)
            print("Total percentage of extra edges found: ", float(total_edges_matched - total_edges_matched)/float(total_edges_edb)*100.0)
            qfs = numpy.array(qfs)
            print("Statistics of q_factors")
            print("Average: ", numpy.average(qfs))
            print("Average: ", numpy.median(qfs))
            print("Stdev: ", numpy.std(qfs))
            print("Min: ", numpy.amin(qfs))
        else:
            print("No edges to match for this set of files.")
        #hist,bin_edges = numpy.histogram(qfs,bins=250)
        #plt.hist(hist,bin_edges)
        #plt.show()

    def test_find_experimental_edge_energies_re_analyze_false(self):
        # Test the keywork re_analyze. The function keeps all edge data from the previous analysis. If re_analyze = False,
        # the function will only change the filtering options to select more or less edges. The filtering should be very
        # fast.
        file_name = os.path.join('Test_Data','DectrisSI.csv')
        
        energies, eels_spectrum = numpy.loadtxt(file_name, delimiter=',',unpack=True)
        energy_step = (energies[-1] - energies[0])/energies.size
        energy_range_ev = numpy.array([energies[0],energies[-1]+energy_step])

        itest=0
        print("Start 10 iterations with analysis.") 
        while itest < 10:
            edge_energies,q_factors = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev)
            print(itest)
            itest+=1
        print('Done.')
        itest=0
        print("Start 10 iterations without analysis.") 
        while itest < 10:
            edge_energies2,q_factors2 = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev,re_analyze=False)
            assert numpy.all(edge_energies == edge_energies2)
            assert numpy.all(q_factors == q_factors2)
            print(itest)
            itest+=1
        print('Done')

    def test_find_experimental_edge_energies_sensitivity(self):
        # Test changes to the sensitivity, which will include less poles as it decreases from 1 to 0. 
        file_name = os.path.join('Test_Data','DectrisSI.csv')

        energies, eels_spectrum = numpy.loadtxt(file_name, delimiter=',',unpack=True)
        energy_step = (energies[-1] - energies[0])/energies.size
        energy_range_ev = numpy.array([energies[0],energies[-1]+energy_step])

        print("Do initial analysis.") 
        # First set sensitivity to 1.0, this will find many edges.
        edge_energies,q_factors = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev,correlation_cutoff_scale=0.0, debug_plotting=False)
        print("Number of edges found = ", edge_energies.size)

        scale=0.0
        # Now change sensitivity parameter from 0 to 1 in steps of 0.1. Number of edges should go from 1 to some max depending on the data and the parameters used (in this case it is 14). 
        print("Start 10 iterations with vaying sensitivity.") 
        while scale <= 1.0:
            edge_energies,q_factors = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev,re_analyze=False,correlation_cutoff_scale=scale,debug_plotting=False)
            print("Sensitivity: ", 1.0-scale)
            print("Number of edges found = ", edge_energies.size)
            print(" ")
            scale = scale + 0.1

if __name__ == '__main__':
    unittest.main()
