# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/core_loss_edge_test.py

import math
import os
import sys
import unittest

import numpy
import scipy.stats

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

from nion.eels_analysis import PeriodicTable 


class TestLibrary(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass

    def test_find_edges_in_interval(self):
        # This test fails in the old version of find_energies_in_interval. Finds only Ti L3 edge.
        ptable=PeriodicTable.PeriodicTable()
        edges=ptable.find_all_edges_in_energy_interval((450.0,470.0))
        # In this interval there should be 7 edges: Ti L2, Ti L3, Nb M1, Ru M3, In M4, Ta N2, and Bi N4
        assert len(edges) == 7
        atomic_numbers = {edge.atomic_number for edge in edges}
        assert atomic_numbers == {22, 41, 44, 49, 73, 83}
        shell_numbers = {edge.shell_number for edge in edges}
        assert shell_numbers == {2, 3, 4}
        subshell_indices = {edge.subshell_index for edge in edges}
        assert subshell_indices == {1, 2, 3, 4}
        shell_strings = {edge.get_shell_str_in_eels_notation(True) for edge in edges}
        assert shell_strings == {"L2", "L3", "M1", "M3", "M4", "N2", "N4"}
        binding_energies = {ptable.nominal_binding_energy_ev(edge) for edge in edges}
        assert binding_energies == {453.8, 460.2, 466.6, 461.4, 451.4, 463.4, 464.0}
        
    def test_find_edges_in_interval_single_atom(self):
        # This tests the new functionality, which allows for filtering by a single atom.
        ptable=PeriodicTable.PeriodicTable()
        edges=ptable.find_all_edges_in_energy_interval((450.0,470.0),22)
        # In this interval there should be 7 edges: Ti L2, Ti L3, Nb M1, Ru M3, In M4, Ta N2, and Bi N4
        assert len(edges) == 2
        atomic_numbers = {edge.atomic_number for edge in edges}
        assert atomic_numbers == {22}
        shell_numbers = {edge.shell_number for edge in edges}
        assert shell_numbers == {2}
        subshell_indices = {edge.subshell_index for edge in edges}
        assert subshell_indices == {2, 3}
        shell_strings = {edge.get_shell_str_in_eels_notation(True) for edge in edges}
        assert shell_strings == {"L2", "L3"}
        binding_energies = {ptable.nominal_binding_energy_ev(edge) for edge in edges}
        assert binding_energies == {453.8, 460.2}

if __name__ == '__main__':
    unittest.main()

#for edge in edges:
#    print(edge.atomic_number, edge.shell_number, edge.subshell_index,edge.get_shell_str_in_eels_notation(True),ptable.nominal_binding_energy_ev(edge))
