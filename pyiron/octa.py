import numpy as np
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from scipy.spatial import cKDTree

class Octa(Atoms):
    def __init__(
        self, structure=None, ref_structure=None, **qwargs
    ):
        self._frame = None
        self._max_dist = None
        self._neigh_site = None
        self._strain = None
        self._neigh_iron = None
        if structure is not None:
            self.structure = structure.copy()
            self.ref_structure = ref_structure
            x = self._get_interstitials()
            super().__init__(
                elements=len(x)*['C'], cell=structure.cell, positions=x, pbc=structure.pbc
            )
            self.center_coordinates_in_unit_cell()
        else:
            super().__init__(**qwargs)

    def _get_interstitials(self):
        neigh = self.structure.get_neighbors(num_neighbors=14)
        cna = self.structure.analyse.pyscal_cna_adaptive(mode='str')
        all_candidates = 0.5 * neigh.vecs[:, 8:] + self.structure.positions[:, None, :]
        all_candidates = all_candidates[cna == 'bcc'].reshape(-1, 3)
        all_candidates = self.structure.analyse.cluster_positions(all_candidates)
        pairs = self.structure.get_neighborhood(all_candidates, num_neighbors=2).indices
        all_candidates = all_candidates[np.all(cna[pairs] == 'bcc', axis=-1)]
        return all_candidates

    @property
    def frame(self):
        if self._frame is None:
            dz = np.diff(
                self.structure.get_neighborhood(self.positions, num_neighbors=2).vecs, axis=-2
            ).squeeze()
            dz = dz[self.neigh_site.flattened.atom_numbers]
            dz /= np.linalg.norm(dz, axis=-1)[:, None]
            dx = self.neigh_site.flattened.vecs
            dx /= np.linalg.norm(dx, axis=-1)[:, None]
            dy = np.cross(dz, dx)
            self._frame = np.stack([dx, dy, dz], axis=-2)
        return self._frame

    def __getitem__(self, key):
        copied = super().__getitem__(key)
        copied.structure = self.structure
        copied.ref_structure = self.ref_structure
        return copied

    @property
    def max_dist(self):
        if self._max_dist is None:
            self._max_dist = np.median(self.get_neighbors(num_neighbors=1).distances)
            self._max_dist *= 0.5 + 1 / np.sqrt(2)
        return self._max_dist

    @property
    def neigh_site(self):
        if self._neigh_site is None:
            self._neigh_site = self.get_neighbors(cutoff_radius=self.max_dist)
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
        eps = np.einsum('naki,nkl,nalj->naij', self._frame, epsilon + self.strain, self._frame)
        eps = eps.reshape(eps.shape[:2]+(9,))
        return eps[:,:,np.array([0, 4, 8, 5, 2, 1])]

    def get_pairs(self):
        return np.stack(
            (self.neigh_site.flattened.atom_numbers, self.neigh_site.flattened.indices)
        , axis=-1)
