#ifndef Fourier_H
#define Fourier_H

#include <cstdlib>
#include <vector>
#include <complex>

using namespace std;

class Fourier{
    public:
        vector< vector< vector< vector<double> > > > get_strain_coeff(
            vector< vector<double> >,
            vector< vector< vector<double> > >,
            vector< vector<double> >,
            vector< vector< vector<double> > >,
            vector< vector<double> >
        );
};
#endif
