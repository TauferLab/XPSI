import math
import numpy as np
import unittest

import knn_angles
import utils

class TestUtils(unittest.TestCase):

    def test_angular_difference(self):
        pi = math.pi
        self.assertAlmostEqual(0, utils.angular_difference([0, 90], [0, 90]), places=6)
        self.assertAlmostEqual(0, utils.angular_difference([45, 12], [45, 12]), places=6)
        self.assertAlmostEqual(0, utils.angular_difference([45, 12], [405, 12]), places=6)
        self.assertAlmostEqual(0, utils.angular_difference([0, 0], [90, 0]), places=6)
        self.assertAlmostEqual(90, utils.angular_difference([0, 90], [90, 90]), places=6)

class TestKNNAngles(unittest.TestCase):
    
    def test__fix_quadrents(self):
        actual = KNN_angles._fix_quadrents(np.array([[[  50,  41], [ -58,  59]],
                                                     [[  41,  50], [  60,  59]],
                                                     [[-160, 161], [ 160, 159]],
                                                     [[ 120, 121], [-179, 170]],
                                                     [[ -39,  40], [-170, 171]],
                                                     [[-171, 172], [-100,  99]]]))


        self.assertTrue(np.array_equal(np.array([[[  50,  41], [ -58,  59]],
                                                 [[  41,  50], [  60,  59]],
                                                 [[-160, 161], [-200, 159]],
                                                 [[-240, 121], [-179, 170]],
                                                 [[ -39,  40], [-170, 171]],
                                                 [[-171, 172], [-100,  99]]]),
                                       actual))
        
    def test__vector_mean(self):
        actual = KNN_angles._vector_mean(np.array([[[90, 1], [-90, 1]],
                                                   [[90, 90], [0, 90]]]))

        #self.assertAlmostEqual(anything, actual[0, 0], places=6)
        self.assertAlmostEqual(0, actual[0, 1], places=6)

        self.assertAlmostEqual(45, actual[1, 0], places=6)
        self.assertAlmostEqual(90, actual[1, 1], places=6)

        
        
if __name__ == '__main__':
    unittest.main()
