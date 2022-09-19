#include "Fourier.h"

vector< vector< vector< vector<double> > > > Fourier::get_strain_coeff(
    vector< vector<double> > km,
    vector< vector< vector<double> > > G,
    vector< vector<double> > g,
    vector< vector< vector<double> > > d,
    vector< vector<double> > x
){
    complex <double> kx(0, 0);
     vector< vector< vector< vector<double > > > > res(int(d.size()), vector< vector< vector<double> > > (int(x.size()), vector< vector<double> > (3, vector<double> (3, 0))));
    vector< vector< complex<double> > > ef(int(d.size()), vector< complex<double> > (int(x.size()), 0));
    vector< vector< complex<double> > > eb(int(d.size()), vector< complex<double> > (int(x.size()), 0));
    for (int K=0; K<int(km.size()); K++)
        for (int R=0; R<int(x.size()); R++)
        {
            kx *= 0;
            for (int i=0; i<3; i++)
                kx += complex<double>(0, -km.at(K).at(i)*x.at(R).at(i));
            ef.at(K).at(R) = exp(kx);
            ef.at(K).at(R) = exp(conj(kx));
        }
    for (int K=0; K<int(km.size()); K++)
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<3; k++)
                    for (int l=0; l<3; l++)
                        for (int r=0; r<int(d.size()); r++)
                            for (int R=0; R<int(x.size()); R++)
                                res.at(r).at(R).at(i).at(j) += km.at(K).at(j)*km.at(K).at(l)*G.at(K).at(i).at(k)*g.at(K).at(r)*d.at(r).at(k).at(l)*real(ef.at(K).at(r)*eb.at(K).at(R));
    return res;
}
