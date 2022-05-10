from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity
from scipy.spatial import Voronoi
from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron
from ase.lattice.cubic import BodyCenteredCubic
import numpy as np
from sklearn.cluster import dbscan


class Dislocation:
    def __init__(
        self,
        lattice_constant=2.855312531,
        repeat_per_angstrom=100,
        repeat_z=4,
        buffer_length=20,
        C_11=1.51830577,
        C_12=0.905516251,
        C_44=0.724588867
    ):
        self.lattice_constant = lattice_constant
        self.repeat_per_angstrom = repeat_per_angstrom
        self.repeat_z = repeat_z
        self.buffer_length = buffer_length
        self.directions = np.array([[1, -2, 1], [1, 0, -1], [1, 1, 1]])
        self._medium = None
        self.C_11 = C_11
        self.C_12 = C_12
        self.C_44 = C_44

    @property
    def _unit_cell(self):
        return ase_to_pyiron(BodyCenteredCubic(
            latticeconstant=self.lattice_constant,
            directions=self.directions,
            symbol='Fe'
        ))

    @property
    def _structure_large_pbc(self):
        unit_cell = self._unit_cell
        return unit_cell.repeat([
            int(self.repeat_per_angstrom / unit_cell.cell[0, 0] + 0.5),
            int(self.repeat_per_angstrom / unit_cell.cell[1, 1] + 0.5),
            self.repeat_z
        ])

    @property
    def _structure_buffered(self):
        structure = self._structure_large_pbc
        structure.cell[0, 0] += 2 * self.buffer_length
        structure.cell[1, 1] += 2 * self.buffer_length
        structure.positions[:, :2] += self.buffer_length
        return structure

    @property
    def burgers_vector(self):
        return np.array([0, 0, np.sqrt(3) * self.lattice_constant / 2])

    @property
    def dislocation_center(self):
        x = self._structure_buffered.positions[:, :2]
        x = x[np.unique(dbscan(x, eps=0.1, min_samples=1)[1], return_index=True)[0]]
        voro = Voronoi(x)
        d = np.linalg.norm(0.5 * self._structure_buffered.cell.diagonal()[:2] - voro.vertices, axis=-1)
        return voro.vertices[np.argmin(d)]

    @property
    def relative_positions(self):
        return self._structure_buffered.positions[:, :2] - self.dislocation_center

    @property
    def elastic_tensor(self):
        C = np.zeros((6, 6))
        C[:3, :3] = self.C_12 + np.eye(3) * (self.C_11 - self.C_12)
        C[3:, 3:] = np.eye(3) * self.C_44
        return C

    @property
    def medium(self):
        medium = LinearElasticity(self.elastic_tensor)
        medium.orientation = self.directions
        return medium

    def get_structure(self):
        structure = self._structure_buffered
        displacements = self.medium.get_dislocation_displacement(
            self.relative_positions, burgers_vector=self.burgers_vector
        )
        structure.positions += displacements
        return structure.center_coordinates_in_unit_cell()
