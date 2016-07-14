# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/Library_test.py

import numpy
import os
import random
import sys
import unittest

# niondata must be available as a module.
# it can be added using something similar to
#   conda dev /path/to/niondata
from nion.data import Calibration
from nion.data import DataAndMetadata

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

# Note: EELSAnalysis is only available in sys.path above is appended with its _parent_ directory.
from EELS import Functions


class TestFunctions(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass

    def test_slow_linear_background(self):
        height, width, depth = 4, 5, 6
        data = numpy.zeros((height, width, depth))
        for row in range(height):
            for column in range(width):
                r1 = random.randint(0, 100)
                r2 = random.randint(0, 100)
                data[row, column, :] = numpy.linspace(r1, r2, depth) + numpy.random.uniform(-1, 1, depth) / 2
        linear_background = Functions.slow_linear_background(data)
        linear_background_subtracted = data - linear_background
        self.assertEqual(linear_background_subtracted.shape, (height, width, depth))
        self.assertTrue(numpy.all(linear_background_subtracted < 1))
        self.assertTrue(numpy.all(linear_background_subtracted > -1))

    def test_stacked_linear_background(self):
        height, width, depth = 4, 5, 6
        data = numpy.zeros((height, width, depth))
        for row in range(height):
            for column in range(width):
                r1 = random.randint(0, 100)
                r2 = random.randint(0, 100)
                data[row, column, :] = numpy.linspace(r1, r2, depth) + numpy.random.uniform(-1, 1, depth) / 2
        linear_background = Functions.stacked_linear_background(data)
        linear_background_subtracted = data - linear_background
        self.assertEqual(linear_background_subtracted.shape, (height, width, depth))
        self.assertTrue(numpy.all(linear_background_subtracted < 1))
        self.assertTrue(numpy.all(linear_background_subtracted > -1))

    def test_subtract_linear_background(self):
        depth, height, width = 80, 6, 8
        data = numpy.zeros((depth, height, width))
        for row in range(height):
            for column in range(width):
                r1 = random.randint(0, 100)
                r2 = random.randint(0, 100)
                data[:, row, column] = numpy.linspace(r1, r2, depth) + numpy.random.uniform(-1, 1, depth) / 2
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(data)
        background_subtracted = Functions.subtract_linear_background(data_and_metadata, (0.0, 1.0), (0.0, 1.0))
        self.assertEqual(background_subtracted.data_shape, (depth, height, width))
        self.assertTrue(numpy.all(numpy.less(background_subtracted.data, 1)))
        self.assertTrue(numpy.all(numpy.greater(background_subtracted.data, -1)))

    def test_subtract_linear_background_with_different_signal_and_fit(self):
        depth, height, width = 80, 6, 8
        data = numpy.zeros((depth, height, width))
        for row in range(height):
            for column in range(width):
                r1 = random.randint(0, 100)
                r2 = random.randint(0, 100)
                data[:, row, column] = numpy.linspace(r1, r2, depth) + numpy.random.uniform(-1, 1, depth) / 2
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(data)
        background_subtracted = Functions.subtract_linear_background(data_and_metadata, (0.1, 0.5), (0.6, 0.9))
        self.assertEqual(background_subtracted.data_shape, (int(depth * 0.3), height, width))
        self.assertTrue(numpy.all(numpy.less(background_subtracted.data[54:72, ...], 1)))
        self.assertTrue(numpy.all(numpy.greater(background_subtracted.data[54:72, ...], -1)))

    def test_signal_and_background_shape_are_consistent_1d(self):
        calibration = Calibration.Calibration(418.92, 0.97, 'eV')
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((2048, ), numpy.float), dimensional_calibrations=[calibration])
        fit_range = 0.2, 0.3
        signal_range = 0.4, 0.5
        signal = Functions.extract_original_signal(data_and_metadata, fit_range, signal_range)
        background = Functions.subtract_background_signal(data_and_metadata, fit_range, signal_range)
        self.assertEqual(signal.data_shape, signal.data.shape)
        self.assertEqual(background.data_shape, background.data.shape)

if __name__ == '__main__':
    unittest.main()