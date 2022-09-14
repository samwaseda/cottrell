from cottrell.pyiron.octa import Octa
import numpy as np
from scipy.sparse import coo_matrix
from pint import UnitRegistry


class Diffusion:
    def __init__(
        self,
        structure,
        bulk,
        dipole_tensor,
        medium,
        force_constants,
        concentration,
        relative_variance=0.5,
        vibration_temperature=1,
        k_point_density=1,
        k_origin=None,
        optimize=True,
    ):
        self.octa = Octa(structure, bulk)
        self._induced_strain = None
        self._phi = None
        self._dipole_tensor = dipole_tensor
        self._dipole_tensor_all = None
        self.medium = medium
        self._force_constants = force_constants
        self._force_constants_all = None
        self.k_point_density = k_point_density
        self._strain_coeff = None
        self.concentration = concentration
        self.relative_variance = relative_variance
        self.vibration_temperature = vibration_temperature
        if k_origin is None:
            k_origin = np.random.random(3) * 0.01
        self.k_origin = k_origin
        self.optimize = optimize
        self._kmesh = None
        self._self_strain = None
        self.ureg = UnitRegistry()
        self.kB = (1 * self.ureg.kelvin * self.ureg.boltzmann_constant).to('eV').magnitude
        self.coeff_octa = np.array([
            -7.98639328e+00, -3.38783856e+00, -3.38766352e+00, -8.68109857e-07,
            3.88493314e-03, 1.31294511e-03, 1.48861535e+01, 1.99343397e+01,
            1.99036546e+01, 2.19714035e+01, -5.88133786e+01, -5.87778330e+01,
            1.95271512e+01, 1.95179820e+01, 2.86738903e+01, 7.29909586e+02,
            1.74032763e+03, 1.78642374e+03, -2.22023335e+03, 5.58965714e+03,
            5.52520510e+03
        ])
        self.coeff_tetra = np.array([
            -6.09270725e+00, -6.09404054e+00, -3.00128245e+00, -4.59663552e-02,
            3.68704147e-02, -8.13361915e-04, -1.38423247e+01, -1.39862582e+01,
            -4.49468634e+01, -3.90335046e+02, -3.90439712e+02, -6.12946382e+01,
            2.09912621e+00, -6.66180141e+00, 2.15156152e+00, 2.12780977e+04,
            2.13970575e+04, 2.97345112e+04, 2.09439450e+05, 2.09624109e+05,
            1.60757026e+04
        ])
        self.E_0 = 0.8153

    @property
    def phi(self):
        if self._phi is None:
            self._phi = self.set_initial_concentration(
                self.concentration, self.relative_variance
            )
        return self._phi

    @phi.setter
    def phi(self, new_phi):
        self._phi = new_phi

    @property
    def vibration_temperature(self):
        return self._vibration_temperature

    @vibration_temperature.setter
    def vibration_temperature(self, t):
        self._vibration_temperature = t
        self._strain_coeff = None

    @property
    def G_k(self):
        return self.medium.get_greens_function(self.kmesh, fourier=True)

    @property
    def strain_coeff(self):
        if self._strain_coeff is None:
            s = np.einsum(
                'Kj,Kl,Kik,Kr,rkl->Krij',
                self.kmesh,
                self.kmesh,
                self.G_k,
                self.gauss_k,
                self.dipole_tensor,
                optimize=self.optimize
            )
            s = np.einsum(
                'Krij,Kr,KR->rRij',
                s,
                np.exp(-1j * np.einsum('kx,rx->kr', self.kmesh, self.octa.octa_positions)),
                np.exp(1j * np.einsum('Kx,Rx->KR', self.kmesh, self.octa.positions)),
                optimize=self.optimize
            )
            self._strain_coeff = np.real(s + np.einsum('rRij->rRji', s)) / self.kspace / 2
        return self._strain_coeff

    @property
    def induced_strain(self):
        strain = np.einsum(
            'rRij,r->Rij',
            self.strain_coeff,
            self.phi,
            optimize=self.optimize
        )
        strain[:len(self.self_strain)] -= self.self_strain
        return strain

    @property
    def self_strain(self):
        if self._self_strain is None:
            self._self_strain = np.real(np.einsum(
                'Kj,Kl,Kik,Rkl,KR->Rij',
                self.kmesh,
                self.kmesh,
                self.G_k,
                self.dipole_tensor,
                self.gauss_k,
                optimize=self.optimize
            )) / self.kspace / 2
            self._self_strain += np.einsum('rij->rji', self._self_strain)
        return np.einsum('r,rij->rij', self.phi, self._self_strain)

    @property
    def dipole_tensor(self):
        if self._dipole_tensor_all is None:
            self._dipole_tensor_all = np.einsum(
                'nik,kl,njl->nij',
                self.octa.octa_frame,
                self._dipole_tensor,
                self.octa.octa_frame,
                optimize=self.optimize
            )
        return self._dipole_tensor_all

    @property
    def force_constants(self):
        if self._force_constants_all is None:
            self._force_constants_all = np.einsum(
                'nik,kl,njl->nij',
                self.octa.octa_frame,
                self._force_constants,
                self.octa.octa_frame,
                optimize=self.optimize
            )
        return self._force_constants_all

    @property
    def psi(self):
        return np.einsum(
            'n,nij->nij',
            self.phi,
            self.dipole_tensor,
        )

    @property
    def gauss_k(self):
        return np.exp(-0.5 * self.vibration_temperature * np.einsum(
            'ki,rij,kj->kr',
            self.kmesh,
            np.linalg.inv(self.force_constants),
            self.kmesh
        ))

    @property
    def kspace(self):
        return self.octa.structure.cell.diagonal().prod()

    @property
    def kmesh(self):
        if self._kmesh is None:
            K_max = np.ones(3) * self.k_point_density
            n_k = np.rint(self.octa.structure.cell.diagonal() * self.k_point_density).astype(int)
            k = [
                np.linspace(0, kk, nn) * 2 * np.pi + ko
                for kk, nn, ko in zip(K_max, n_k, self.k_origin)
            ]
            self._kmesh = np.concatenate(np.meshgrid(*k)).reshape(3, -1).T
        return self._kmesh

    def set_initial_concentration(self, concentration, relative_variance=0):
        concentration *= len(self.octa.structure) / len(self.octa.octa_positions)
        variance = relative_variance * concentration
        variance = variance * (2 * np.random.random(len(self.octa.octa_positions)) - 1)
        variance -= np.mean(variance)
        return concentration * np.ones(len(self.octa.octa_positions)) + variance

    def _get_energy(self, induced_strain=False):
        if induced_strain:
            strain = self.octa.get_eps(epsilon=self.induced_strain)
        else:
            strain = self.octa.get_eps()
        E = np.zeros(len(strain))
        E[:len(self.octa.octa_positions)] = eps_polynom(
            strain[:len(self.octa.octa_positions)], coeff=self.coeff_octa
        )
        E[len(self.octa.octa_positions):] = eps_polynom(
            strain[len(self.octa.octa_positions):], coeff=self.coeff_tetra
        ) + self.E_0
        return E

    @property
    def chemical_interactions(self):
        E_all = np.zeros(len(self.octa))
        np.add.at(
            E_all,
            self.octa.chem_neigh['atom_numbers'],
            get_chemical_interactions(
                self.octa.chem_neigh['dist']
            ) * self.phi[self.octa.chem_neigh['indices']]
        )
        return E_all

    def get_transition_tensor(self, temperature, induced_strain=False, E_min=0, kappa=1.0e13):
        E_all = self._get_energy(induced_strain) + self.chemical_interactions
        E = -E_all[self.octa.pairs_double[:, 1]]
        E += np.tile(E_all[len(self.octa.octa_positions):], 2)
        K = kappa * np.exp(-E / (self.kB * temperature))
        K = coo_matrix(
            (K, (*self.octa.pairs_double.T,)),
            shape=(len(self.phi), len(self.phi))
        )
        return K

    @property
    def order_parameter(self):
        Z = np.sum(np.absolute(self.octa.frame[:, 0, -1]) * self.phi[:, None], axis=0)
        return np.sqrt(3 / 2 * (np.sum(Z**2) / self.phi.sum()**2 - 1 / 3))


def get_chemical_interactions(d, c_0=3.3591338560479134, alpha=0.741257004198575):
    return c_0 * np.exp(-alpha * d**2)


def eps_polynom(eps, coeff):
    eps_tmp = np.array(eps).reshape(-1, 6)
    eps_tmp = np.concatenate((
        eps_tmp,
        eps_tmp**2,
        eps_tmp[:, :3] * np.roll(eps_tmp[:, :3], 1, axis=-1),
        eps_tmp**4
    ), axis=-1)
    return (np.einsum('i,ni->n', coeff, eps_tmp)).reshape(np.asarray(eps).shape[:-1])
