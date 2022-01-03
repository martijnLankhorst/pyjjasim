from josephson_circuit import Circuit
from static_problem import StaticProblem
from time_evolution import TimeEvolutionProblem, DEF_TEMPERATURE_PROFILE, AnnealingProblem2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

def find_extreme_vortex_configuration(array: Circuit, Is_base, f, start_lambda=0.5, W=10,
                                      dt=2.0, interval_count=1000, interval_steps=20,
                                      vortex_mobility=0.001, start_T=1, T_factor=1.03,
                                      lambda_factor=0.8, rerun_at_max_lambda=False, _maxtries=3):
    """
    Attempts to find a vortex configuration which allows the largest value of lambda
    """
    Is_func = lambda x: Is_base * x
    f_func = lambda x: f

    print(f"searching at lambda={start_lambda}")
    out_lambda = np.zeros(0, dtype=np.double)
    out_n = np.zeros((array.face_count(), 0), dtype=int)
    cur_f = f_func(start_lambda)
    cur_Is = Is_func(start_lambda)

    # main step
    prob = AnnealingProblem2(array, time_step=dt, interval_steps=interval_steps, interval_count=interval_count,
                             vortex_mobility=vortex_mobility, frustration=cur_f,
                             current_sources=cur_Is, problem_count=W, start_T=start_T, T_factor=T_factor)
    vortex_configurations, energies, status, configurations, T = prob.compute()
    vortex_configurations = vortex_configurations[:, status==0]
    Wp = np.sum(status==0)
    if np.sum(status==0) == 0:
        if _maxtries > 1:
            l, n = find_extreme_vortex_configuration(array, Is_base, f, start_lambda=start_lambda * lambda_factor, W=W,
                                          dt=dt, interval_count=interval_count, interval_steps=interval_steps,
                                          vortex_mobility=vortex_mobility, start_T=start_T, T_factor=T_factor,
                                          lambda_factor=lambda_factor, _maxtries=_maxtries-1, rerun_at_max_lambda=False)
            out_lambda = np.append(out_lambda, l)
            out_n = np.concatenate((out_n, n), axis=1)
    else:
        lambdas = np.zeros(Wp)
        confs = np.zeros(Wp, dtype=object)
        for i in range(Wp):
            n = vortex_configurations[:, i]
            m_prob = StaticProblem(array, vortex_configuration=n, frustration=f_func(0))
            lambdas[i], _, confs[i], _ = m_prob.compute_maximal_parameter(Is_func=Is_func, f_func=f_func)
        mask = ~np.isnan(lambdas)
        out_lambda = np.append(out_lambda, lambdas[mask])
        out_n = np.concatenate((out_n, vortex_configurations[:, mask]), axis=1)
        if sum(mask) == 0:
            if _maxtries > 1:
                l, n = find_extreme_vortex_configuration(array, Is_base, f, start_lambda=start_lambda * lambda_factor, W=W,
                                                         dt=dt, interval_count=interval_count, interval_steps=interval_steps,
                                                         vortex_mobility=vortex_mobility, start_T=start_T, T_factor=T_factor,
                                                         lambda_factor=lambda_factor, _maxtries=_maxtries-1, rerun_at_max_lambda=False)
                out_lambda = np.append(out_lambda, l)
                out_n = np.concatenate((out_n, n), axis=1)
    sorter = np.argsort(out_lambda)[::-1]
    out_lambda, out_n = out_lambda[sorter], out_n[:, sorter]
    if rerun_at_max_lambda:
        l, n = find_extreme_vortex_configuration(array, Is_base, f, start_lambda=out_lambda[0], W=W,
                                                 dt=dt, interval_count=interval_count, interval_steps=interval_steps,
                                                 vortex_mobility=vortex_mobility, start_T=start_T, T_factor=T_factor,
                                                 lambda_factor=lambda_factor, _maxtries=_maxtries, rerun_at_max_lambda=False)
        out_lambda = np.append(out_lambda, l)
        out_n = np.concatenate((out_n, n), axis=1)
        sorter = np.argsort(out_lambda)[::-1]
        out_lambda, out_n = out_lambda[sorter], out_n[:, sorter]
    return out_lambda, out_n
