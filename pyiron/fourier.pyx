# distutils: language = c++

from Fourier cimport Fourier as Fouriercpp
from libcpp.vector cimport vector
import numpy as np

cdef class Fourier:
    cdef Fouriercpp c_fourier

    def get_strain_coeff(self, km, G, g, d, x):
        return self.c_fourier.get_strain_coeff(km, G, g, d, x)

