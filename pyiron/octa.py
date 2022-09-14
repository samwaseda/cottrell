import numpy as np
from pyiron_atomistics.atomistics.structure.atoms import Atoms


class Octa(Atoms):
    def __init__(
        self, structure=None, ref_structure=None, **qwargs
    ):
        self._frame = None
        self._max_dist = None
        self._neigh_site = None
        self._strain = None
        self._lattice_constant = None
        self._pairs = None
        self._pairs_double = None
        self.r_chemical_cutoff = 10
        self._chem_neigh = None
        if structure is not None:
            self.structure = structure.copy()
            self.ref_structure = ref_structure
            x = self._get_interstitials()
            super().__init__(
                elements=len(x) * ['C'], cell=structure.cell, positions=x, pbc=structure.pbc
            )
            self.center_coordinates_in_unit_cell()
            x = self._get_tetrahedral()
            self += Atoms(positions=x, cell=self.cell, elements=len(x) * ['H'])
        else:
            super().__init__(**qwargs)

    @property
    def lattice_constant(self):
        if self._lattice_constant is None:
            neigh = self.ref_structure.get_neighbors(num_neighbors=8)
            self._lattice_constant = np.median(neigh.distances) / np.sqrt(3) * 2
        return self._lattice_constant

    @property
    def _first_shell_cutoff(self):
        return self.lattice_constant * (np.sqrt(2) + 1) / 4

    def _get_tetrahedral(self):
        neigh = self.get_neighbors(cutoff_radius=self._first_shell_cutoff)
        x = self.positions[neigh.flattened.atom_numbers] + 0.5 * neigh.flattened.vecs
        x = self.get_wrapped_coordinates(x)
        return self.analyse.cluster_positions(x, eps=0.2)

    @property
    def _is_bcc(self):
        neigh = self.structure.get_neighbors(num_neighbors=8)
        return neigh.get_steinhardt_parameter(4) > 0.45

    def _get_interstitials(self):
        neigh = self.structure.get_neighbors(num_neighbors=14)
        all_candidates = 0.5 * neigh.vecs[:, 8:] + self.structure.positions[:, None, :]
        all_candidates = all_candidates[self._is_bcc].reshape(-1, 3)
        all_candidates = self.structure.analyse.cluster_positions(all_candidates)
        pairs = self.structure.get_neighborhood(all_candidates, num_neighbors=2).indices
        all_candidates = all_candidates[np.all(self._is_bcc[pairs], axis=-1)]
        return all_candidates

    @property
    def frame(self):
        if self._frame is None:
            neigh = self.get_neighbors(cutoff_radius=self._first_shell_cutoff * 0.5)
            v = self.structure.get_neighborhood(
                self.positions[self.select_index('C')], num_neighbors=2
            ).vecs
            octa_axis = np.diff(v, axis=-2).squeeze()
            octa_second = neigh.vecs[self.select_index('C'), 0]
            octa_third = np.cross(octa_axis, octa_second)
            octa_frame = np.einsum('ijk->jki', [octa_axis, octa_second, octa_third])
            tetra_axis = neigh.vecs[self.select_index('H'), 0]
            tetra_second = octa_axis[neigh.indices[self.select_index('H'), 0]]
            tetra_third = np.cross(tetra_axis, tetra_second)
            tetra_frame = np.einsum('ijk->jki', [tetra_second, tetra_third, tetra_axis])
            self._frame = np.concatenate((octa_frame, tetra_frame), axis=0)
            self._frame /= np.linalg.norm(self._frame, axis=-2)[:, None, :]
            if np.any(np.linalg.det(self._frame) < 0.99):
                raise ValueError('Frame not orthonormal')
        return self._frame

    @property
    def octa_frame(self):
        return self.frame[self.select_index('C')]

    @property
    def tetra_frame(self):
        return self.frame[self.select_index('H')]

    def __getitem__(self, key):
        copied = super().__getitem__(key)
        copied.structure = self.structure
        copied.ref_structure = self.ref_structure.copy()
        return copied

    @property
    def strain(self):
        if self._strain is None:
            strain = self.structure.analyse.get_strain(self.ref_structure, 8)
            self._strain = np.concatenate((
                np.mean(strain[self.structure.get_neighborhood(
                    self.octa_positions, num_neighbors=2
                ).indices], axis=1),
                np.mean(strain[self.structure.get_neighborhood(
                    self.tetra_positions, num_neighbors=4
                ).indices], axis=1)
            ), axis=0)
        return self._strain

    def get_eps(self, epsilon=np.zeros((3, 3))):
        eps = np.einsum('nki,nkl,nlj->nij', self.frame, epsilon + self.strain, self.frame)
        eps = eps.reshape(len(eps), 9)
        return eps[:, np.array([0, 4, 8, 5, 2, 1])]

    @property
    def pairs(self):
        if self._pairs is None:
            neigh = self.get_neighbors(num_neighbors=2)
            self._pairs = neigh.indices[self.select_index('H')]
        return self._pairs

    @property
    def pairs_double(self):
        if self._pairs_double is None:
            self._pairs_double = np.concatenate((self.pairs, self.pairs[:, ::-1]), axis=0)
        return self._pairs_double

    @property
    def chem_neigh(self):
        if self._chem_neigh is None:
            neigh = self.get_neighborhood(
                self.positions,
                num_neighbors=None,
                cutoff_radius=self.r_chemical_cutoff
            )
            cond = neigh.flattened.indices < len(self.octa_positions)
            self._chem_neigh = {
                'dist': neigh.flattened.distances[cond],
                'atom_numbers': neigh.flattened.atom_numbers[cond],
                'indices': neigh.flattened.indices[cond]
            }
        return self._chem_neigh

    @property
    def tetra_positions(self):
        return self.positions[self.select_index('H')]

    @property
    def octa_positions(self):
        return self.positions[self.select_index('C')]
