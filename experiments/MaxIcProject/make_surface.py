from multiprocessing import Pool

import scipy.interpolate
from matplotlib.colors import LogNorm
from scipy.interpolate import NearestNDInterpolator, interp2d

from experiments.MaxIcProject.generate_ensembles import load, ensemble, compute_single
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from static_problem import *
from josephson_circuit import *

import pyamg

def E_surface(f, n, f_out, I_out, N):
    x = np.sqrt(np.arange(0, N)/(N-1))
    F = f + np.array(f_out - f)[:, None] * x
    I = np.array(I_out)[:, None] * x
    array = SquareArray(4, 4)
    h = array.horizontal_junctions()
    energies = np.zeros(F.shape)
    N = F.shape[1]
    W = F.shape[0]
    for w in range(W):
        Is = h * I[w, 0]
        f = F[w, 0]
        prob = StaticProblem(array, vortex_configuration=n, frustration=f, current_sources=Is)
        out = prob.compute()
        for i in range(N):
            Is = h * I[w, i]
            f = F[w, i]
            prob = StaticProblem(array, vortex _configuration=n, frustration=f, current_sources=Is)
            out = prob.compute(initial_guess=out[0])
            # print(out[1])
            energies[w, i] = np.mean(out[0].get_Etot())
    return F, I, energies

def E_surfaces(f, n, f_out, I_out, N):
    M = len(f)
    W = f_out.shape[0]
    Fs = np.zeros((W, N, M))
    Is = np.zeros((W, N, M))
    Es = np.zeros((W, N, M))
    for k in range(M):
        print(k)
        Fs[..., k], Is[..., k], Es[..., k] = E_surface(f[k], n[:, k], f_out[:, k], I_out[:, k] / 4, N=N)
    return Fs, Is, Es

def E_surfaces2(f, n, f_out, I_out, N):
    M = len(f)
    W = f_out.shape[0]
    Fs = np.zeros((W, N, M))
    Is = np.zeros((W, N, M))
    Es = np.zeros((W, N, M))
    with Pool(processes=8) as pool:
        out = pool.starmap(E_surface, [(f[k], n[:, k], f_out[:, k], I_out[:, k] / 4, N) for k in range(M)] )
    for k in range(M):
        Fs[..., k], Is[..., k], Es[..., k] = out[k]
    return Fs, Is, Es


def gen_E_surfaces(fn, N):
    f, n, f_out, I_out = load(fn)
    Fs, Is, Es = E_surfaces2(f, n, f_out, I_out, N)
    filename = fn[:-4] + "_E.npy"
    with open(filename, 'wb') as ffile:
        np.save(ffile, Fs)
        np.save(ffile, Is)
        np.save(ffile, Es)

def load_E_surfaces(fn):
    filename = fn[:-4] + "_E.npy"
    with open(filename, 'rb') as ffile:
        Fs = np.load(ffile)
        Is = np.load(ffile)
        Es = np.load(ffile)
        return Fs, Is, Es

def ccw(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) > ( by - ay) * (cx - ax)

def is_in(F, I, f, i):
    dF = 0.5 * (F[1:] + F[:-1]) - f[..., None]
    dI = 0.5 * (I[1:] + I[:-1]) - i[..., None]
    # dF = F[1:] - f[..., None]
    # dI = I[1:] - i[..., None]
    closest = np.argmin(dF*dF + dI*dI, axis=-1)
    print(closest.shape)
    return (F[closest+1] - F[closest]) * (i - I[closest]) - (I[closest+1] - I[closest]) * (f - F[closest]) > 0


def is_in2(F, I, f, i, fp):
    d = (((f[..., None] - fp) * I) - i[..., None] * (F - fp))
    print(d)
    left = np.argmax((d > 0).astype(int), axis=-1)
    right = left - 1
    D = (F[left] - F[right]) * (0 - i) - (I[left] - I[right]) * (fp - f)
    Fp = ((F[left]*I[right] - F[right]*I[left]) * (fp - f) - (F[left] - F[right]) * (fp * i)) / D
    Ip = ((F[left]*I[right] - F[right]*I[left]) * (0 - i)  - (I[left] - I[right]) * (fp * i)) / D
    dist1 = (Fp - fp) * (Fp - fp) + Ip * Ip
    dist2 = (f - fp) * (f - fp) + i * i
    return dist1 > dist2

def is_in3(F, I, f, i):
    return rayintersect2(f, i, F, I)


from collections import namedtuple
from pprint import pprint as pp
import sys

Pt = namedtuple('Pt', 'x, y')  # Point
Edge = namedtuple('Edge', 'a, b')  # Polygon edge from a to b
Poly = namedtuple('Poly', 'name, edges')  # Polygon

_eps = 0.00001
_huge = sys.float_info.max
_tiny = sys.float_info.min


def rayintersectseg(p, edge):
    ''' takes a point p=Pt() and an edge of two endpoints a,b=Pt() of a line segment returns boolean
    '''
    a, b = edge
    if a.y > b.y:
        a, b = b, a
    if p.y == a.y or p.y == b.y:
        p = Pt(p.x, p.y + _eps)

    intersect = False

    if (p.y > b.y or p.y < a.y) or (
            p.x > max(a.x, b.x)):
        return False

    if p.x < min(a.x, b.x):
        intersect = True
    else:
        if abs(a.x - b.x) > _tiny:
            m_red = (b.y - a.y) / float(b.x - a.x)
        else:
            m_red = _huge
        if abs(a.x - p.x) > _tiny:
            m_blue = (p.y - a.y) / float(p.x - a.x)
        else:
            m_blue = _huge
        intersect = m_blue >= m_red
    return intersect

def rayintersect2(px, py, poly_x, poly_y):
    """
    px, py:         (M,)
    poly_x, poly_y  (n,)
    out: (M,)
    """
    px, py = np.array(px), np.array(py)
    poly_x, poly_y = np.array(poly_x), np.array(poly_y)
    if poly_x.ndim != 1:
        raise ValueError("poly_x, y must be 1d array")
    if poly_x.shape != poly_y.shape:
        raise ValueError("poly_x, y must be 1d array")
    if len(poly_x) < 2:
        raise ValueError("polygon must contain atleast two points")
    if px.size != py.size:
        raise ValueError("px, py must be same size")
    p_shape = px.shape
    px, py = px.ravel(), py.ravel()
    point_count = len(px)
    poly_len = len(poly_x)
    bx, by = np.roll(poly_x, -1), np.roll(poly_y, -1)
    mask = poly_y > by
    ax, bx = np.where(mask, bx, poly_x), np.where(mask, poly_x, bx)
    ay, by = np.where(mask, by, poly_y), np.where(mask, poly_y, by)
    intersect = np.zeros((point_count, poly_len), dtype=bool)
    px = px[:, None] * np.ones((1, poly_len))
    py = py[:, None] * np.ones((1, poly_len))
    py += _eps * ((py > ay) | (py > by))
    p_mask = px < np.minimum(ax, bx)
    intersect[p_mask] = True
    m_red = np.ones(poly_len) * _huge
    m_blue = np.ones((point_count, poly_len)) * _huge
    mask = np.abs(ax - bx) > _tiny
    m_red[mask] = (by - ay)[mask] / (bx - ax)[mask]
    mask = np.abs(ax - px) > _tiny
    m_blue[mask] = (py - ay)[mask] / (px - ax)[mask]
    intersect[~p_mask] = (m_blue >= m_red)[~p_mask]
    intersect[((py > by) | (py < ay)) | (px > np.maximum(ax, bx))] = False
    return (np.sum(intersect.astype(int), axis=1) % 2 == 1).reshape(p_shape)

def _odd(x): return x % 2 == 1


def ispointinside(p, poly):
    ln = len(poly)
    return _odd(sum(rayintersectseg(p, edge)
                    for edge in poly.edges))


def polypp(poly):
    print("\n  Polygon(name='%s', edges=(" % poly.name)
    print('   ', ',\n    '.join(str(e) for e in poly.edges) + '\n    ))')


# if __name__ == '__main__':
#     polys = [
#         Poly(name='square', edges=(
#             Edge(a=Pt(x=0, y=0), b=Pt(x=10, y=0)),
#             Edge(a=Pt(x=10, y=0), b=Pt(x=10, y=10)),
#             Edge(a=Pt(x=10, y=10), b=Pt(x=0, y=10)),
#             Edge(a=Pt(x=0, y=10), b=Pt(x=0, y=0))
#         )),
#         Poly(name='square_hole', edges=(
#             Edge(a=Pt(x=0, y=0), b=Pt(x=10, y=0)),
#             Edge(a=Pt(x=10, y=0), b=Pt(x=10, y=10)),
#             Edge(a=Pt(x=10, y=10), b=Pt(x=0, y=10)),
#             Edge(a=Pt(x=0, y=10), b=Pt(x=0, y=0)),
#             Edge(a=Pt(x=2.5, y=2.5), b=Pt(x=7.5, y=2.5)),
#             Edge(a=Pt(x=7.5, y=2.5), b=Pt(x=7.5, y=7.5)),
#             Edge(a=Pt(x=7.5, y=7.5), b=Pt(x=2.5, y=7.5)),
#             Edge(a=Pt(x=2.5, y=7.5), b=Pt(x=2.5, y=2.5))
#         )),
#         Poly(name='strange', edges=(
#             Edge(a=Pt(x=0, y=0), b=Pt(x=2.5, y=2.5)),
#             Edge(a=Pt(x=2.5, y=2.5), b=Pt(x=0, y=10)),
#             Edge(a=Pt(x=0, y=10), b=Pt(x=2.5, y=7.5)),
#             Edge(a=Pt(x=2.5, y=7.5), b=Pt(x=7.5, y=7.5)),
#             Edge(a=Pt(x=7.5, y=7.5), b=Pt(x=10, y=10)),
#             Edge(a=Pt(x=10, y=10), b=Pt(x=10, y=0)),
#             Edge(a=Pt(x=10, y=0), b=Pt(x=2.5, y=2.5))
#         )),
#         Poly(name='exagon', edges=(
#             Edge(a=Pt(x=3, y=0), b=Pt(x=7, y=0)),
#             Edge(a=Pt(x=7, y=0), b=Pt(x=10, y=5)),
#             Edge(a=Pt(x=10, y=5), b=Pt(x=7, y=10)),
#             Edge(a=Pt(x=7, y=10), b=Pt(x=3, y=10)),
#             Edge(a=Pt(x=3, y=10), b=Pt(x=0, y=5)),
#             Edge(a=Pt(x=0, y=5), b=Pt(x=3, y=0))
#         )),
#     ]
#     testpoints = (Pt(x=5, y=5), Pt(x=5, y=8),
#                   Pt(x=-10, y=5), Pt(x=0, y=5),
#                   Pt(x=10, y=5), Pt(x=8, y=5),
#                   Pt(x=10, y=10))
#
#     print("\n TESTING WHETHER POINTS ARE WITHIN POLYGONS")
#     for poly in polys:
#         polypp(poly)
#         print('   ', '\t'.join("%s: %s" % (p, ispointinside(p, poly))
#                                for p in testpoints[:3]))
#         print('   ', '\t'.join("%s: %s" % (p, ispointinside(p, poly))
#                                for p in testpoints[3:6]))
#         print('   ', '\t'.join("%s: %s" % (p, ispointinside(p, poly))
#                                for p in testpoints[6:]))


if __name__ == "__main__":
    # a = np.array([2, 3, 4])
    # b = np.array([3, 2, 5])
    # a, b = np.sort((a, b), axis=0)
    # print(a, b)
    # px = [0.1, 1.1]
    # py = [0.1, 0.1]
    # poly_x = [0, 1, 1, 0]
    # poly_y = [0, 0, 1, 1]
    # print(rayintersect2(px, py, poly_x, poly_y))

    fn = "sqN5_complete_ensemble__num_angles121_betaL_0_D.npy"
    N = 5
    Q = (N-1) ** 2
    f, n, f_out, I_out = load(fn)

    P = 1000
    F = np.linspace(0, 0.5, P)
    I = np.linspace(0, 5, P)
    FF, II = np.meshgrid(F, I)
    count = np.zeros((P, P))
    fig, ax = plt.subplots()
    M = f_out.shape[1]

    for k in range(M):
        print(k)
        count += is_in3(f_out[:, k], I_out[:, k], FF, II)

    m = ax.pcolormesh(FF, II, np.log(count) / np.log(2) / Q,  cmap='viridis')
    fig.colorbar(m, ax=ax)
    # plt.plot(f_out[:, k], I_out[:, k])
    plt.show()
# ax = plt.axes(projection='3d')
# fn = "sqN4_complete_ensemble__num_angles121_betaL_0_B.npy"
# gen_E_surfaces(fn, 3)
# Fs, Is, Es = load_E_surfaces(fn)
# for k in range(Fs.shape[2]):
#     ax.plot_surface(Fs[..., k], Is[..., k], Es[..., k], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# plt.show()