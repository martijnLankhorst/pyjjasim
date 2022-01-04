from functools import partial
from multiprocessing import Pool

import numpy as np

from find_extreme_vortex_configuration import find_extreme_vortex_configuration
from josephson_circuit import Circuit, SquareArray
from static_problem import StaticProblem, DefaultCPR


class VortexConfigurationEnsemble:

    def __init__(self, array: Circuit, frustrations, vortex_configurations):
        """
        array                       Array object
        frustrations                (ensemble_size,) array
        vortex_configurations       (face_count, ensemble_size,) array
        """
        self.array = array
        self.frustrations = np.array(frustrations, dtype=np.double).ravel()
        self.vortex_configurations = np.array(vortex_configurations, dtype=int)
        if self.vortex_configurations.shape != (self.get_array().face_count(), self.frustrations.size):
            raise ValueError("vortex_configurations must have shape (face_count, ensemble_size,)")
        self.vortex_configurations, idx = np.unique(self.vortex_configurations, axis=1, return_index=True)
        self.frustrations = self.frustrations[idx]

    def get_array(self):
        return self.array

    def get_frustrations(self):
        return self.frustrations

    def get_vortex_configurations(self):
        return self.vortex_configurations

    def ensemble_size(self):
        return len(self.frustrations)

    def get_frustration_bounds(self, title=None):
        W = self.ensemble_size()
        array =self.get_array()
        print(W)
        f_low, f_high = np.zeros(W), np.zeros(W)
        f, n = self.frustrations, self.vortex_configurations
        with Pool(processes=16) as pool:
            out = pool.starmap(compute_frust_bnd_func, [(array, f[i], n[:, i]) for i in range(W)])
        for i, o in enumerate(out):
            f_low[i], f_high[i] = o
            print(i, f_low[i], f_high[i])
        mask = np.isnan(f_low) | np.isnan(f_high)
        f_low = f_low[~mask]
        f_high = f_high[~mask]
        ns = self.vortex_configurations[:, ~mask]
        if title is not None:
            with open(title, 'wb') as ffile:
                np.save(ffile, f_low)
                np.save(ffile, f_high)
                np.save(ffile, ns)
        return f_low, f_high, ns

    def compute_maximal_current(self, current_sources_base, num_frustration_points, f_low=None, f_high=None, ns=None):
        if f_low is None:
            f_low, f_high, ns = self.get_frustration_bounds()
        f = np.linspace(np.min(f_low), np.max(f_high), num_frustration_points)
        max_current_factors = np.zeros(num_frustration_points)
        array = self.get_array()
        for i, fp in enumerate(f):
            print(i)
            ids = np.flatnonzero((fp > f_low) & (fp < f_high))
            max_I_factor = 0
            print(ids)
            print(max_I_factor)
            for k in range(1 + ((len(ids)-1) // 16)):
                batch = np.arange(16 * k, min(16 * k + 16, len(ids)))
                print(len(ids), batch)
                if len(batch) > 0:
                    with Pool(processes=16) as pool:
                        out = pool.starmap(compute_max_I_func, [(array, fp, ns[:, i], current_sources_base,
                                                                 max_I_factor) for i in ids[batch]])
                    print([o for o in out])
                    max_I_factor = max(max_I_factor, np.max(np.array([0] + [o for o in out])))
                    print(max_I_factor)
            max_current_factors[i] = max_I_factor
        return f, max_current_factors

    # def compute_max_energy_without_current_sources(self, num_frustration_points):
    #     f_low, f_high = self.get_frustration_bounds()
    #     f = np.linspace(np.min(f_low), np.max(f_high), num_frustration_points)
    #     max_energy = np.zeros(num_frustration_points)
    #     for i, fp in enumerate(f):
    #         energies = [0]
    #         for j, (fl, fh) in enumerate(zip(f_low, f_high)):
    #             n = self.vortex_configurations[:, j]
    #             if fp > fl and fp < fh:
    #                 prob = StaticProblem(self.get_array(), vortex_configuration=n,
    #                                      frustration=fp)
    #                 config, status, info = prob.compute()
    #                 if status == 0:
    #                     energies += [np.mean(config.get_Etot())]
    #         max_energy[i] = np.max(energies)
    #     return f, max_energy


def generate_vortex_ensemble_with_high_current_sources(array, Is_base, f_list, start_lambda=0.2,
                                                       W=30, dt=2.0, interval_count=1000, interval_steps=20,
                                                       vortex_mobility=0.001, start_T=1.0, T_factor=1.03,
                                                       lambda_factor=0.8, rerun_at_max_lambda=True, title=None):
    with Pool(processes=16) as pool:
        out = pool.starmap(find_extreme_vortex_configuration, [(array, Is_base, f, start_lambda, W, dt,
                                                                interval_count, interval_steps, vortex_mobility,
                                                                start_T, T_factor, lambda_factor,
                                                                rerun_at_max_lambda) for f in f_list])
    ns = [o[1][:, :max(len(o[0]), 5)] for o in out]
    fs = np.repeat(f_list, [o.shape[1] for o in ns])
    ns = np.concatenate(ns, axis=1)
    if title is not None:
        with open(title, 'wb') as ffile:
            np.save(ffile, np.array([start_lambda, W, dt, interval_count, interval_steps,
                                     vortex_mobility, start_T, T_factor, lambda_factor]))
            np.save(ffile, fs)
            np.save(ffile, ns)
    return fs, ns


def compute_frust_bnd_func(array, f, n):
    out, _, _ =  StaticProblem(array, vortex_configuration=n).compute_frustration_bounds(start_frustration=f)
    return out

def compute_max_I_func(array, f, n, Is_base, upper):
    out, _, _, _ = StaticProblem(array, vortex_configuration=n, current_sources=Is_base,
                                 frustration=f).compute_maximal_current(abort_if_below=upper)
    return out