# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/Library_test.py

import numpy
import os
import random
import sys
import unittest
import numpy
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
        file_name = os.path.join('Test_Data','DectrisSI.csv')

        energies, eels_spectrum = numpy.loadtxt(file_name, delimiter=',',unpack=True)
        energy_step = (energies[-1] - energies[0])/energies.size
        energy_range_ev = numpy.array([energies[0],energies[-1]+energy_step])

        print("Do initial analysis.") 
        # First set sensitivity to 1.0, this will find many edges.
        edge_energies,q_factors = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev, debug_plotting=True)
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
        outF = open("test.dat","w")
        total_edges_found = 0
        total_edges_edb = 0
        total_edges_matched = 0
        qfs=[]
        mintol=0.1
        for file_name in file_names:
           # Load data from text file.
            data_file = glob.glob('./EELS_Atlas_Major/**/' + file_name,recursive = True)[0]
            #data_file = glob.glob('./EELS_Atlas_All/**/' + file_name,recursive = True)[0]
            
            edge_file = glob.glob('./EELS_Atlas_Major/**/' + 'edges_' + os.path.splitext(file_name)[0]+'.dat', recursive=True)[0]
            #edge_file = glob.glob('./EELS_Atlas_All/**/' + 'edges_' + os.path.splitext(file_name)[0]+'.dat', recursive=True)[0]
            energies, eels_spectrum = numpy.loadtxt(data_file, delimiter=',',unpack=True)
            energy_step = (energies[-1] - energies[0])/energies.size
            energy_range_ev = numpy.array([energies[0],energies[-1]+energy_step])

            if estart[i_search] < 0:
                estart[i_search] = tmp1[i_search]

            search_range = [estart[i_search],efin[i_search]]
            print(file_name,':')
            edge_energies,q_factors = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev,search_range)
            print("Edges found at the following energies:")
            outF.write(file_name)
            outF.write("\n\n")
            df=pandas.read_csv(edge_file, delim_whitespace=True, header=None)
            edge_names, edb_edge_energies = df.to_numpy().T
            match=[False]*edb_edge_energies.size
            matched=[False]*edge_energies.size
            print("Edges matched:")
            outF.write("Edges matched:\n")
            print("Edge\tfound\tatlas")
            outF.write("Edge\tfound\tatlas\n")
            i_edb = 0
            total_edges_found = total_edges_found + edge_energies.size
            all_matched = True
            for edb_edge_energy in edb_edge_energies:
                if edb_edge_energy > search_range[0] and edb_edge_energy < search_range[1]:
                    total_edges_edb = total_edges_edb + 1 
                    
                    edb_edge_found = False
                    i=0
                    ibest=0
                    for energy in edge_energies:
                        if True: #not matched[i]:
                            if (abs(edb_edge_energy - energy)/edb_edge_energy < 0.1) or abs(edb_edge_energy-energy) < 10.0:
                                if True: #not match[i_edb]:
                                    print("\t".join((str(edge_names[i_edb]),str(int(energy)),str(edb_edge_energy),str(q_factors[i]))))
                                    outF.write("\t".join((str(edge_names[i_edb]),str(int(energy)),str(edb_edge_energy),str(q_factors[i]))))
                                    outF.write("\n")
                                    match[i_edb]=True
                                    matched[i]=True
                                    qfs = qfs + [q_factors[i]]
                        i=i+1
                    if match[i_edb]:
                        total_edges_matched += 1
                    all_matched = all_matched and match[i_edb]
                i_edb += 1
                
            i=0
            print("Extra edges found:")
            outF.write("Extra edges found:\n")
            for energy in edge_energies:
                if not matched[i]:
                    print('\t'.join((str(int(energy)),str(q_factors[i]))))
                    outF.write('\t'.join((str(int(energy)),str(q_factors[i]))))
                    outF.write('\n')
                i += 1
                
            outF.write("\n")
            if not all_matched:
                print("Edges not matched:")
                outF.write("Edges not matched: " + file_name + "\n")
            
            i=0
            for energy in edb_edge_energies:
                if (not match[i]) and (energy < search_range[1]) and (energy > search_range[0]) :
                    print('\t'.join((str(edge_names[i]), str(int(energy)))))
                    outF.write('\t'.join((str(edge_names[i]), str(int(energy)))))
                    outF.write('\n')
                i += 1
            outF.write("\n")
            outF.write("\n")
            print(" ")
            print(" ")
            i_search=i_search+1

        print("Percentage of edges found: ", float(total_edges_matched)/float(total_edges_edb)*100.0)
        print("Percentage of extra edges found: ", float(total_edges_found - total_edges_matched)/float(total_edges_edb)*100.0)
        qfs = numpy.array(qfs)
        print("Statistics of q_factors")
        print("Average: ", numpy.average(qfs))
        print("Average: ", numpy.median(qfs))
        print("Stdev: ", numpy.std(qfs))
        print("Min: ", numpy.amin(qfs))
        #hist,bin_edges = numpy.histogram(qfs,bins=250)
        #plt.hist(hist,bin_edges)
        #plt.show()
        outF.close()

    def test_find_experimental_edge_energies_re_analyze_false(self):

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
        file_name = os.path.join('Test_Data','DectrisSI.csv')

        energies, eels_spectrum = numpy.loadtxt(file_name, delimiter=',',unpack=True)
        energy_step = (energies[-1] - energies[0])/energies.size
        energy_range_ev = numpy.array([energies[0],energies[-1]+energy_step])

        print("Do initial analysis.") 
        # First set sensitivity to 1.0, this will find many edges.
        edge_energies,q_factors = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev,correlation_cutoff_scale=0.0, debug_plotting=False)
        print("Number of edges found = ", edge_energies.size)

        sens=1.0
        # Now change sensitivity parameter from 1 to zero in steps of 0.1
        print("Start 10 iterations with vaying sensitivity.") 
        while sens >= 0.0:
            edge_energies,q_factors = EELS_DataAnalysis.find_experimental_edge_energies(eels_spectrum, energy_range_ev,re_analyze=False,correlation_cutoff_scale=sens,debug_plotting=False)
            print("Sensitivity: ", sens)
            print("Number of edges found = ", edge_energies.size)
            print(" ")
            sens = sens - 0.1

if __name__ == '__main__':
    unittest.main()
