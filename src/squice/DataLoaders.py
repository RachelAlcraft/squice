"""
Module to load different data fromats from file and convert into a numpy matrix
"""

from abc import ABC, abstractmethod
import numpy as np


####################################################################################
class DataLoader(ABC):
    """Abstract class to load data from file and convert to a numpy matrix"""

    def __init__(self, filename):
        self.filename = filename
        self.mtx = None

    @abstractmethod
    def load(self):
        pass


####################################################################################
class NumpyFile(DataLoader):
    """Class to load a file of numerical data from numpy serislisation format"""

    def load(self):
        """Load the numpy data"""
        self.mtx = np.load(self.filename)


####################################################################################
class ElectronDensityCcp4(DataLoader):
    """Class to load electron density in the ccp4 format"""

    def load(self):
        """Load the numpy data"""
        self.mtx = None


####################################################################################

if __name__ == "__main__":
    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) #shape (3x3)
    # np.save('speed_data.npy', a)
    npf = NumpyFile("speed_data.npy")
    npf.load()
