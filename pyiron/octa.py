import numpy as np
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from scipy.spatial import cKDTree

class Octa(Atoms):
    def __init__(self, structure=None, ref_structure=None, min_steinhardt_parameter=0.755, **qwargs):
        self._frame = None
        self._cond = None
        self._max_dist = None
        self._neigh_site = None
        self._strain = None
        self._neigh_iron = None
        if structure is not None:
            self.structure = structure.copy()
            self.ref_structure = ref_structure
            interstitials = structure.analyse.get_interstitials(num_neighbors=6, variance_buffer=100)
            x = interstitials.positions[interstitials.get_steinhardt_parameters(4)>min_steinhardt_parameter]
            super().__init__(
                elements=len(x)*['C'], cell=structure.cell, positions=x, pbc=structure.pbc
            )
            self.center_coordinates_in_unit_cell()
        else:
            super().__init__(**qwargs)

    @property
    def cond(self):
        if self._cond is None:
            self._cond = self.neigh_site.distances < self.max_dist
            self._cond *= np.absolute(np.linalg.det(self.frame)) > 0.5
        return self._cond

    @property
    def frame(self):
        if self._frame is None:
            z = np.diff(self.neigh_iron.vecs, axis=-2).squeeze()
            y = z[self.neigh_site.indices]
            z = np.ones_like(self.neigh_site.vecs)*z[:,None,:]
            x = self.neigh_site.vecs
            self._frame = np.concatenate((x, y, z), axis=-1).reshape(x.shape+(3,))
            self._frame = np.einsum('nkij,nki->nkij', self._frame, 1/np.linalg.norm(self._frame, axis=-1))
            self._frame[:,:,-1] = np.cross(self._frame[:,:,0], self._frame[:,:,1], axisa=-1, axisb=-1)
        return self._frame

    def __getitem__(self, key):
        copied = super().__getitem__(key)
        copied.structure = self.structure
        copied.ref_structure = self.ref_structure
        return copied

    @property
    def max_dist(self):
        if self._max_dist is None:
            lattice_constant = self.ref_structure.get_neighbors(num_neighbors=1).distances.min()
            lattice_constant *= 2/np.sqrt(3)
            self._max_dist = (1+np.sqrt(2))/4*lattice_constant
        return self._max_dist

    @property
    def neigh_site(self):
        if self._neigh_site is None:
            self._neigh_site = self.get_neighbors(num_neighbors=4)
        return self._neigh_site

    @property
    def neigh_iron(self):
        if self._neigh_iron is None:
            self._neigh_iron = self.structure.get_neighborhood(self.positions, num_neighbors=2)
        return self._neigh_iron

    @property
    def strain(self):
        if self._strain is None:
            strain = self.structure.analyse.get_strain(self.ref_structure, 8)
            self._strain = np.mean(strain[self.neigh_iron.indices], axis=1)
        return self._strain

    def get_eps(self, epsilon=np.zeros((3,3))):
        eps = np.einsum('naki,nkl,nalj->naij', self._frame, epsilon+self.strain, self._frame)
        eps = eps.reshape(eps.shape[:2]+(9,))
        return eps[:,:,np.array([0, 4, 8, 5, 2, 1])]

    def get_pairs(self):
        return np.stack((np.where(self.cond)[0], self.neigh_site.indices[self.cond]), axis=-1)
