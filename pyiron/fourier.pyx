# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "Fourier.cpp":
    pass

cdef extern from "Fourier.h":
    cdef cppclass Fourier:
        Fourier() except +
        vector[vector[vector[vector[double]]]] get_strain_coeff(
            vector[vector[double]],
            vector[vector[vector[double]]],
            vector[vector[double]],
            vector[vector[vector[double]]],
            vector[vector[double]]
        )

