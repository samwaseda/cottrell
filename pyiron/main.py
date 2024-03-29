from pyiron_atomistics.atomistics.structure.factory import StructureFactory
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_continuum.elasticity.linear_elasticity import LinearElasticity
from cottrell.pyiron.diffusion import Diffusion
import numpy as np
from pyiron_base.jobs.job.generic import GenericJob
from pyiron_base.storage.datacontainer import DataContainer
from collections import defaultdict
from tqdm import tqdm
from pandas import DataFrame


class FeCAdatom(GenericJob):  # Create a custom job class
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = DataContainer(table_name='input')
        self.output = DataContainer(table_name='output')
        self.input['kappa'] = 1e13
        self.input['dt'] = 1e-6
        self.input['n_outer_loops'] = 10000
        self.input['n_inner_loops'] = 10
        self.input['n_print'] = 1
        self.input['lattice_parameter'] = 2.87
        self.input['C_11'] = 1.51830577
        self.input['C_12'] = 0.905516251
        self.input['C_44'] = 0.724588867
        self.input['orientation'] = np.eye(3)
        self.input['dipole_tensor'] = np.array([8.03096531, 3.40030112, 3.40030112]) * np.eye(3)
        self.input['force_constants'] = np.array(
            [17.60546082, 10.03435345, 10.03435345]
        ) * np.eye(3)
        self.input['concentration'] = 0.01
        self.input['relative_variance'] = 0.5
        self.input['temperature'] = 300
        self.input['minimum_energy'] = 0.0
        self.input['vibration_temperature'] = 1
        self.input['k_point_density'] = 1
        self.input['max_flow'] = 0.1
        self._diffusion = None
        self._medium = None
        self.output['order_parameter'] = []
        self.output['time'] = []
        self.output['std'] =  []
        self.output['phi'] = []

    @property
    def elastic_tensor(self):
        C = np.zeros((6, 6))
        C[:3, :3] = self.input['C_12']+np.eye(3)*(self.input['C_11']-self.input['C_12'])
        C[3:, 3:] = np.eye(3)*self.input['C_44']
        return C

    @property
    def medium(self):
        if self._medium is None:
            self._medium = LinearElasticity(self.elastic_tensor)
            self._medium.orientation = self.input['orientation']
        return self._medium

    @property
    def diffusion(self):
        if self._diffusion is None:
            basis = StructureFactory().bulk('Fe', a=self.input['lattice_parameter'], cubic=True)
            self._diffusion = Diffusion(
                structure=self.structure,
                bulk=basis,
                dipole_tensor=self.input['dipole_tensor'],
                medium=self.medium,
                force_constants=self.input['force_constants'],
                concentration=self.input['concentration'],
                relative_variance=self.input['relative_variance'],
                vibration_temperature=self.input['vibration_temperature'],
                k_point_density=self.input['k_point_density'],
            )
        return self._diffusion

    def run_adatom(self):
        dt = self.input['dt']
        acc_time = 0
        for i_cycle in tqdm(range(self.input['n_outer_loops'])):
            temperature = self.input['temperature']
            K = self.diffusion.get_transition_tensor(
                temperature,
                induced_strain=True,
                E_min=self.input['minimum_energy'],
            )
            for _ in range(self.input['n_inner_loops']):
                source = self.diffusion.phi
                availability = 1 - self.diffusion.phi
                dphi = availability * K.T.dot(source) - source * K.dot(availability)
                dt *= 1.1
                while (
                    np.any(-dt*dphi/self.input['max_flow'] > self.diffusion.phi)
                    or np.any(1-dphi*dt/self.input['max_flow'] < self.diffusion.phi)
                ):
                    dt /= 2
                self.diffusion.phi += dt*dphi
                acc_time += dt
            if i_cycle % self.input['n_print'] == 0:
                self.output['order_parameter'].append(self.diffusion.order_parameter)
                self.output['time'].append(acc_time)
                self.output['std'].append(np.std(self.diffusion.phi))
                self.output['phi'].append(self.diffusion.phi.copy())

    def run_static(self):
        if self.diffusion.phi is None:
            self.set_initial_concentration()
        self.run_adatom()
        self.status.finished = True

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(hdf=self.project_hdf5)
        self.output.to_hdf(hdf=self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(hdf=self.project_hdf5)
        self.output.from_hdf(hdf=self.project_hdf5)

    def write_input(self):
        pass


def get_potential():
    potential = {}
    potential['Config'] = [['pair_style eam/alloy\n', 'pair_coeff * * Fe-C-Bec07.eam Fe C\n']]
    potential['Filename'] = [['/cmmc/u/samsstud/local/pyiron_local/pyiron_local/potentials/Fe-C-Bec07.eam']]
    potential['Model'] = ['EAM']
    potential['Name'] = ['Raulot']
    potential['Species'] = [['Fe', 'C']]
    potential = DataFrame(potential)
    return potential

