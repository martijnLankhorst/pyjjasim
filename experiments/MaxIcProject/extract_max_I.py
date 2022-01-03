
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from experiments.MaxIcProject.generate_ensembles import ensemble

matplotlib.use("TkAgg")

def extract_max_I(f_out, I_out, f_probe):
    I = []
    for f in f_probe:
        angle_nr, prob_nr = np.nonzero(((f_out[:-1, :] >= f) & (f_out[1:, :] < f)) | ((f_out[:-1, :] <= f) & (f_out[1:, :] > f)))
        f1, f2 = f_out[angle_nr, prob_nr], f_out[angle_nr+1, prob_nr]
        I1, I2 = I_out[angle_nr, prob_nr], I_out[angle_nr+1, prob_nr]
        I += [np.max(((f2 - f) * I1 + (f - f1) * I2) / (f2 - f1))]
    return np.array(I)


N = 3
A = 121
P = (N-1) ** 2
f, n = ensemble(N)
S = len(f)
print(S)
titles = ('sqN3_complete_ensemble__num_angles121_A.npy',
          'sqN3_complete_ensemble__num_angles121_betaL_1_A.npy',
          'sqN3_complete_ensemble__num_angles121_betaL_5_A.npy')


for title in titles:
    with open(title, 'rb') as ffile:
        f_ens = np.load(ffile)
        n_ens = np.load(ffile)
        f_out = np.load(ffile)
        I_out = np.load(ffile)


    F = np.linspace(0, 0.5, 501)
    I = extract_max_I(f_out, I_out, F)
    plt.plot(F, I/N)
    plt.plot(1 - F, I/N)
plt.show()

