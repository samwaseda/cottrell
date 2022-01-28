from pyiron_mpie.adatom.octa import Octa
import numpy as np
from scipy.sparse import coo_matrix

class Diffusion:
    def __init__(
        self,
        structure,
        bulk,
        dipole_tensor,
        medium,
        force_constants,
        vibration_temperature=1,
        k_points=20,
        k_origin=None,
        optimize=True,
    ):
        self.octa = Octa(structure, bulk)
        self._induced_strain = None
        self.phi = None
        self.chemical_interactions = {}
        self._dipole_tensor = dipole_tensor
        self.medium = medium
        self._force_constants = force_constants
        self.k_points = k_points
        self.vibration_temperature = vibration_temperature
        if k_origin is None:
            k_origin = np.random.random(3)*0.01
        self.k_origin = k_origin
        self.optimize = optimize
        self._kmesh = None
        self._psi_k_coeff = None
        self._self_strain = None

    @property
    def vibration_temperature(self):
        return self._vibration_temperature

    @vibration_temperature.setter
    def vibration_temperature(self, t):
        self._vibration_temperature = t
        self._psi_k_coeff = None

    @property
    def G_k(self):
        return self.medium.get_greens_function(self.kmesh, fourier=True)

    @property
    def induced_strain(self):
        strain = np.real(np.einsum(
            'Kj,Kl,Kik,Kkl,KR->Rij',
            self.kmesh,
            self.kmesh,
            self.G_k,
            self.psi_k,
            np.exp(1j*np.einsum('Kx,Rx->KR', self.kmesh, self.octa.positions)),
            optimize=self.optimize
        ))/self.kspace
        return 0.5*(strain+np.einsum('Rij->Rji', strain))-self.self_strain

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
            ))/self.kspace
            self._self_strain = 0.5*(self._self_strain+np.einsum('Rij->Rji', self._self_strain))
        return np.einsum('R,Rij->Rij', self.phi, self._self_strain)

    @property
    def dipole_tensor(self):
        return np.einsum(
            'nik,kl,njl->nij',
            self.octa.frame[:,0],
            self._dipole_tensor,
            self.octa.frame[:,0],
            optimize=self.optimize
        )

    @property
    def force_constants(self):
        return np.einsum(
            'nik,kl,njl->nij',
            self.octa.frame[:,0],
            self._force_constants,
            self.octa.frame[:,0],
            optimize=self.optimize
        )

    @property
    def psi(self):
        return np.einsum(
            'n,nij->nij',
            self.phi,
            self.dipole_tensor,
        )

    @property
    def gauss_k(self):
        return np.exp(-0.5*self.vibration_temperature*np.einsum(
            'ki,rij,kj->kr',
            self.kmesh,
            np.linalg.inv(self.force_constants),
            self.kmesh
        ))

    @property
    def psi_k(self):
        if self._psi_k_coeff is None:
            self._psi_k_coeff = np.einsum(
                'kr,kr,rij->krij',
                self.gauss_k,
                np.exp(-1j*np.einsum('kx,rx->kr', self.kmesh, self.octa.positions)),
                self.dipole_tensor,
                optimize=self.optimize
            )
        return np.einsum('r,krij->kij', self.phi, self._psi_k_coeff)

    @property
    def kspace(self):
        return self.octa.structure.cell.diagonal().prod()

    @property
    def kmesh(self):
        if self._kmesh is None:
            K_max = self.k_points/self.octa.structure.cell.diagonal()
            k = np.linspace(0, K_max, self.k_points)*2*np.pi+self.k_origin
            self._kmesh = np.concatenate(np.meshgrid(*k.T)).reshape(3, -1).T
        return self._kmesh

    def set_initial_concentration(self, concentration, relative_variance=0):
        concentration *= len(self.octa.structure)/len(self.octa)
        variance = relative_variance*concentration
        variance = variance*(2*np.random.random(len(self.octa))-1)
        variance -= np.mean(variance)
        self.phi = concentration*np.ones(len(self.octa))+variance

    def _get_energy(self, induced_strain=False, E_min=0):
        if induced_strain:
            E = eps_polynom(self.octa.get_eps(epsilon=self.induced_strain))
        else:
            E = eps_polynom(self.octa.get_eps())
        E[E<E_min] = E_min
        return E

    def get_transition_tensor(self, temperature, induced_strain=False, E_min=0, kappa=1.0e13):
        pairs = self.octa.get_pairs()
        K = kappa*np.exp(-self._get_energy(induced_strain, E_min)/(8.617e-5*temperature))
        K = coo_matrix(
            (K[self.octa.cond], (pairs[:,0], pairs[:,1])),
            shape=(len(self.phi), len(self.phi))
        )
        return K

    def get_availability_tensor(self, temperature):
        D = np.ones_like(self.phi)
        kBT = 8.617e-5*temperature
        np.multiply.at(
            D,
            self.chemical_interactions['pairs'][:,0],
            1-self.phi[self.chemical_interactions['pairs'][:,1]]*(
                1-np.exp(-self.chemical_interactions['values']/kBT)
            )
        )
        return D

    def set_chemical_interactions(self, pairs, values):
        self.chemical_interactions['pairs'] = pairs
        self.chemical_interactions['values'] = values

    @property
    def order_parameter(self):
        Z = np.sum(np.absolute(self.octa.frame[:,0,-1])*self.phi[:,None], axis=0)
        return np.sqrt(3/2*(np.sum(Z**2)/self.phi.sum()**2-1/3))


coeff = np.array([
    8.17984994e-01, -4.14176065e-01, -2.89418590e+00, 1.89796295e+00, 
    -1.54401897e-01, 3.44244054e-02, -2.42412993e-02, -4.97767144e+01, 
    -1.69775655e+01, -2.30522494e+00, -1.13675776e+02, -4.66512355e+02, 
    -6.23934680e+02, -2.42675859e+01, -3.22719965e+01, -1.65944800e+01, 
    9.24620586e-01, 4.43084706e+03, -6.98243642e+02, 2.38722525e+06, 
    1.21321689e+06, 1.80753892e+06
])

def eps_polynom(eps, coeff=coeff):
    eps_tmp = np.array(eps).reshape(-1, 6)
    eps_tmp = np.concatenate((
        eps_tmp,
        eps_tmp**2,
        eps_tmp[:,:3]*np.roll(eps_tmp[:,:3], 1, axis=-1),
        eps_tmp**4
    ), axis=-1)
    return (np.einsum('i,ni->n', coeff[1:], eps_tmp)+coeff[0]).reshape(np.asarray(eps).shape[:-1])
