from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, LogNorm
import matplotlib.animation as animation
import matplotlib.cm as cm
matplotlib.use("TkAgg")

from pyJJAsim.static_problem import StaticConfiguration
from pyJJAsim.time_evolution import TimeEvolutionResult

__all__ = ["Plot", "CircuitPlot", "CircuitMovie"]


class Plot:

    """Plots a circuit configuration.

    Allows one to show node quantities, junction quantities, face quantities and vortices.
        - Node quantities are shown as colors on the nodes
        - junction quantities are displayed by arrows whose length is proportional to the quantity value
        - face quantities are shown as colors on the faces
        - vortices are displayed by symbols (concentric rings, vorticity equals nr of rings, color shows sign).\

    Base class used by both CircuitPlot and CircuitMovie.\

    Node quantity options:
        - "phi": gauge depent phases
        - "U": node voltage or potential

    Junction quantity options:
        - "theta": gauge_invariant_phase_difference
        - "V": voltage
        - "I": current
        - "I_sup": supercurrent
        - "I_s": current sources
        - "EJ": josephson energy
        - "EM": magnetic_energy
        - "EC": capacitive energy
        - "Etot": total_energy

    Face quantity options:
        - "Phi: magnetic_flux
        - "n": vortex configuration
        - "J": face_current


    """
    def __init__(self, config: StaticConfiguration | TimeEvolutionResult, time_point=0, show_vortices=True,
                 vortex_diameter=0.25, vortex_color=(0, 0, 0),
                 anti_vortex_color=(0.8, 0.1, 0.2), vortex_alpha=1, show_grid=True, grid_width=1,
                 grid_color=(0.3, 0.5, 0.9), grid_alpha=0.5, show_colorbar=True, show_arrows=True,
                 arrow_quantity="I", arrow_width=0.005, arrow_scale=1, arrow_headwidth=3, arrow_headlength=5,
                 arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1, arrow_color=(0.2, 0.4, 0.7),
                 arrow_alpha=1,  show_nodes=True, node_diameter=0.2,
                 node_face_color=(1,1,1), node_edge_color=(0, 0, 0), node_alpha=1, show_node_quantity=False,
                 node_quantity="phase", node_quantity_cmap=None, node_quantity_clim=(0, 1), node_quantity_alpha=1,
                 node_quantity_logarithmic_colors=False, show_face_quantity=False, face_quantity="n",
                 face_quantity_cmap=None, face_quantity_clim=(0, 1), face_quantity_alpha=1,
                 face_quantity_logarithmic_colors=False,
                 figsize=None, title=""):

        """
        Constructor for Plot and handling plot options.
        """
        self.config = config
        self.time_point = time_point

        if not isinstance(config, (StaticConfiguration, TimeEvolutionResult)):
            raise ValueError("config must be a StaticConfiguration or DynamicConfiguration object")

        self.show_vortices = show_vortices
        self.vortex_diameter = vortex_diameter
        self.vortex_color = vortex_color
        self.anti_vortex_color = anti_vortex_color
        self.vortex_alpha = vortex_alpha
        self.show_grid = show_grid
        self.grid_width = grid_width
        self.grid_color = grid_color
        self.grid_alpha = grid_alpha
        self.show_colorbar = show_colorbar
        self.show_arrows = show_arrows
        self.arrow_quantity = arrow_quantity

        self.arrow_width = arrow_width
        self.arrow_scale = arrow_scale
        self.arrow_headwidth = arrow_headwidth
        self.arrow_headlength = arrow_headlength
        self.arrow_headaxislength = arrow_headaxislength
        self.arrow_minshaft = arrow_minshaft
        self.arrow_minlength = arrow_minlength

        self.arrow_color = arrow_color
        self.arrow_alpha = arrow_alpha
        self.show_nodes = show_nodes
        self.node_diameter = node_diameter
        self.node_face_color = node_face_color
        self.node_edge_color = node_edge_color

        self.node_alpha = node_alpha
        self.show_node_quantity = show_node_quantity
        self.node_quantity = node_quantity
        self.node_quantity_cmap = node_quantity_cmap
        self.node_quantity_clim = node_quantity_clim
        self.node_quantity_alpha = node_quantity_alpha
        self.node_quantity_logarithmic_colors = node_quantity_logarithmic_colors

        self.show_face_quantity = show_face_quantity
        self.face_quantity = face_quantity
        self.face_quantity_cmap = face_quantity_cmap
        self.face_quantity_clim = face_quantity_clim
        self.face_quantity_alpha = face_quantity_alpha
        self.face_quantity_logarithmic_colors = face_quantity_logarithmic_colors

        self.figsize = figsize if figsize is not None else [6.4, 4.8]
        self.colorbar = None
        self.title = title

        self.fig = None
        self.ax = None

    _node_quantities = {
        "phi": 0, "phase": 0, "phases": 0, "U": 1, "potential": 1,
    }

    _junction_quantities = {
        "th": 0, "theta": 0, "phase_difference": 0,  "gauge_invariant_phase_difference": 0,
        "V": 1, "voltage": 1,
        "I": 3, "current": 3,
        "Isup": 4,  "I_sup": 4, "Isuper": 4, "I_super": 4, "supercurrent": 4, "super_current": 4,
        "I_s": 5, "Is": 5, "current_sources": 5,
        "EJ": 6, "josephson_energy": 6, "EM": 7, "magnetic_energy": 7, "EC": 8, "capacitive_energy": 8,
        "capacitance_energy": 8, "Etot": 9, "E_tot": 9, "ETot": 9, "total_energy": 9, "energy": 9,
    }

    _face_quantities = {
        "Phi": 0, "flux": 0, "magnetic_flux": 0,
        "n": 2, "vortices": 2, "vortex_configuration": 2, "face_current": 3, "J": 3,
    }

    def _get_lims(self):
        x, y = self.config.get_circuit().get_node_coordinates()
        xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
        dx, dy = xmax - xmin, ymax - ymin
        D = self.node_diameter * 0.5
        return xmin - 0.05 * dx - D, xmax + 0.05 * dx + D, ymin - 0.05 * dy - D, ymax + 0.05 * dy + D

    def _set_axes(self, ):
        xmin, xmax, ymin, ymax = self._get_lims()
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.time_label = self.ax.annotate("", (xmin + 0.99 * (xmax - xmin), ymin + 0.98 * (ymax - ymin)), color=[1, 0.5, 0.2], ha='right', va='center')
        x0, y0, width, height = self.ax.get_position().bounds
        a_width = width * self.figsize[0]
        a_height = height * self.figsize[1]
        if a_width / (xmax - xmin) > a_height / (ymax - ymin):
            new_width = a_height * (xmax - xmin) / (ymax - ymin) / self.figsize[0]
            x0 = x0 + (width - new_width) / 2
            width = new_width
        else:
            new_height = a_width * (ymax - ymin) / (xmax - xmin) / self.figsize[1]
            y0 = y0 + (height - new_height) / 2
            height = new_height
        self.ax.set_position([x0, y0, width, height])

    def _is_static(self):
        return isinstance(self.config, StaticConfiguration)

    def _get_node_quantity(self):
        """
        Get node quantity (either "phi": gauge dependent phases or  "U": potential)
        """
        if isinstance(self.node_quantity, np.ndarray):
            return self.node_quantity.flatten()
        quantity = self._node_quantities[self.node_quantity]
        if quantity == 0:   # phi
            out = self.config.get_phi() if self._is_static() else self.config.get_phi(self.time_point)
            out = out.copy()
            out -= np.round(out / (np.pi * 2.0)).astype(out.dtype) * np.pi * 2.0
            return out
        if quantity == 1:   # U
            return self.config.get_U(self.time_point)

    def _get_junction_quantity(self):
        if isinstance(self.arrow_quantity, np.ndarray):
            return self.arrow_quantity.flatten()
        quantity = self._junction_quantities[self.arrow_quantity]
        if quantity == 0:   # theta
            out = self.config.get_theta() if self._is_static() else self.config.get_theta(self.time_point)
            out = out.copy()
            out -= np.round(out / (np.pi * 2.0)).astype(out.dtype) * np.pi * 2.0
            return out
        if quantity == 1:   # V
            return self.config.get_V(self.time_point)
        if quantity == 3:   # I
            return self.config.get_I() if self._is_static() else self.config.get_I(self.time_point)
        if quantity == 4:   # I
            return self.config.get_I() if self._is_static() else self.config.get_Isup(self.time_point)
        if quantity == 5:   # I_ext_J
            return self.config.problem._Is() if self._is_static() else self.config.problem._Is(self.time_point)
        if quantity == 6:   # EJ
            return self.config.get_EJ() if self._is_static() else self.config.get_EJ(self.time_point)
        if quantity == 7:   # EM
            return self.config.get_EM() if self._is_static() else self.config.get_EM(self.time_point)
        if quantity == 8:   # EC
            return self.config.get_EC(self.time_point)
        if quantity == 9:   # Etot
            return self.config.get_Etot() if self._is_static() else self.config.get_Etot(self.time_point)

    def _get_face_quantity(self):
        if isinstance(self.face_quantity, np.ndarray):
            return self.face_quantity.flatten()
        quantity = self._face_quantities[self.face_quantity]
        if quantity == 0:   # Phi
            return self.config.get_flux() if self._is_static() else self.config.get_flux(self.time_point)
        if quantity == 2:   # n
            return self.config.get_n() if self._is_static() else self.config.get_n(self.time_point)
        if quantity == 3:   # J
            return self.config.get_J() if self._is_static() else self.config.get_J(self.time_point)

    def _marker_scale_factor(self):
        xlim = self.ax.get_xlim()
        x0, y0, width, height = self.ax.get_position().bounds
        return (width * self.figsize[0]) / (xlim[1] - xlim[0]) *72


    def _plot_grid(self):
        x1, y1, x2, y2 = self.config.get_circuit().get_juncion_coordinates()
        self.ax.plot(np.stack((x1, x2)), np.stack((y1, y2)), color=self.grid_color,
                     alpha=self.grid_alpha, linewidth=self.grid_width, zorder=0)

    def _plot_nodes(self, node_quantity):
        x, y = self.config.get_circuit().get_node_coordinates()
        marker_size = self.node_diameter * self._marker_scale_factor()
        if not self.show_node_quantity:
            nodes_handle = self.ax.plot([x], [y], markeredgecolor=self.node_edge_color, markerfacecolor=self.node_face_color,
                                        markersize=marker_size, marker="o", alpha=self.node_alpha, zorder=2)
            nodes_handle = nodes_handle[0]
        else:
            cnorm = Normalize(*self.node_quantity_clim) if not self.node_quantity_logarithmic_colors else LogNorm(*self.node_quantity_clim)
            nodes_handle = self.ax.scatter(x.flatten(), y.flatten(), s=marker_size**2, c=node_quantity, cmap=self.node_quantity_cmap,
                                           edgecolors=self.node_edge_color, alpha=self.node_quantity_alpha, norm=cnorm)
            if self.show_colorbar:
                label = self.node_quantity if isinstance(self.node_quantity, str) else ""
                self.colorbar = plt.colorbar(cm.ScalarMappable(norm=cnorm, cmap=self.node_quantity_cmap), ax=self.ax, label=label)
        return nodes_handle

    def _plot_arrows(self, arrow_quantity):
        I = arrow_quantity * self.arrow_scale
        x1, y1, x2, y2 = self.config.get_circuit().get_juncion_coordinates()
        xq = x1 + 0.5 * (1 - I) * (x2 - x1)
        yq = y1 + 0.5 * (1 - I) * (y2 - y1)
        dxq, dyq = I * (x2 - x1), I * (y2 - y1)
        return self.ax.quiver(xq, yq, dxq, dyq, edgecolor=self.arrow_color, facecolor=self.arrow_color,
                   angles='xy', scale=1, scale_units='xy', width=self.arrow_width,
                   headwidth=self.arrow_headwidth, headlength=self.arrow_headlength,
                   headaxislength=self.arrow_headaxislength, minshaft=self.arrow_minshaft,
                   minlength=self.arrow_minlength, alpha=self.arrow_alpha, zorder=3)

    def _plot_faces(self, face_quantity):
        nodes = self.config.get_circuit().get_faces()
        x, y = self.config.get_circuit().get_node_coordinates()
        verts = [np.stack((x[n], y[n]), axis=-1) for n in nodes]
        cnorm = Normalize(*self.face_quantity_clim) if not self.face_quantity_logarithmic_colors else LogNorm(*self.face_quantity_clim)
        coll = PolyCollection(verts, array=face_quantity, edgecolors='none', cmap=self.face_quantity_cmap,
                              norm=cnorm, alpha=self.face_quantity_alpha, zorder=-1)
        faces_handle = self.ax.add_collection(coll)
        if self.show_colorbar and not (self.show_nodes and self.show_node_quantity):
            label = self.face_quantity if isinstance(self.face_quantity, str) else ""
            self.colorbar = plt.colorbar(cm.ScalarMappable(norm=cnorm, cmap=self.face_quantity_cmap), ax=self.ax, label=label)
        return faces_handle

    def _plot_vortices(self, n):
        marker_size = self.vortex_diameter * self._marker_scale_factor()
        xc, yc = self.config.get_circuit().get_face_centroids()
        ns = np.unique(n)
        vort_handles = []
        for ni in ns:
            if ni != 0:
                color = self.vortex_color if ni > 0 else self.anti_vortex_color
                na = np.abs(ni)
                for k in reversed(range(na)):
                    frac = (2 * k + 1) / (2 * na - 1)
                    p = self.ax.plot(xc[n == ni], yc[n == ni], markerfacecolor=color,
                                     markeredgecolor=color, marker="o", linestyle="",
                                     markersize=frac * marker_size, alpha=self.vortex_alpha, zorder=4)
                    vort_handles += p
                    if k > 0:
                        frac = (2 * k) / (2 * na - 1)
                        p = self.ax.plot(xc[n == ni], yc[n == ni], markerfacecolor=[1, 1, 1],
                                         markeredgecolor=[1, 1, 1], marker="o", linestyle="",
                                         markersize=frac * marker_size, alpha=self.vortex_alpha, zorder=4)
                        vort_handles += p
        return vort_handles

    def test_method(self):
        pass

class CircuitPlot(Plot):

    def __init__(self, config: StaticConfiguration | TimeEvolutionResult, time_point=0, show_vortices=True,
                 vortex_diameter=0.25, vortex_color=(0, 0, 0), anti_vortex_color=(0.8, 0.1, 0.2), vortex_alpha=1,
                 show_grid=True, grid_width=1, grid_color=(0.3, 0.5, 0.9), grid_alpha=0.5, show_colorbar=True,
                 show_arrows=True, arrow_quantity="I", arrow_width=0.005, arrow_scale=1, arrow_headwidth=3,
                 arrow_headlength=5, arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1,
                 arrow_color=(0.2, 0.4, 0.7), arrow_alpha=1, show_nodes=True, node_diameter=0.2,
                 node_face_color=(1, 1, 1), node_edge_color=(0, 0, 0), node_alpha=1, show_node_quantity=False,
                 node_quantity="phase", node_quantity_cmap=None, node_quantity_clim=(0, 1), node_quantity_alpha=1,
                 node_quantity_logarithmic_colors=False, show_face_quantity=False, face_quantity="n",
                 face_quantity_cmap=None, face_quantity_clim=(0, 1), face_quantity_alpha=1,
                 face_quantity_logarithmic_colors=False, figsize=None, title=""):

        super().__init__(config, time_point, show_vortices, vortex_diameter, vortex_color, anti_vortex_color,
                         vortex_alpha, show_grid, grid_width, grid_color, grid_alpha, show_colorbar, show_arrows,
                         arrow_quantity, arrow_width, arrow_scale, arrow_headwidth, arrow_headlength,
                         arrow_headaxislength, arrow_minshaft, arrow_minlength, arrow_color, arrow_alpha, show_nodes,
                         node_diameter, node_face_color, node_edge_color, node_alpha, show_node_quantity, node_quantity,
                         node_quantity_cmap, node_quantity_clim, node_quantity_alpha, node_quantity_logarithmic_colors,
                         show_face_quantity, face_quantity, face_quantity_cmap, face_quantity_clim, face_quantity_alpha,
                         face_quantity_logarithmic_colors, figsize, title)

        self.time_point = time_point

        if isinstance(config, TimeEvolutionResult):
            if not config.problem.store_time_steps[self.time_point]:
                raise ValueError("The requested timepoint from config to plot has not been stored during "
                                 "simulation (set with the config.store_time_steps property)")

    @staticmethod
    def _assert_single_configuration(data):
        if data.ndim >= 2:
            raise ValueError("must select single configuration")

    def make(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        plt.title(self.title)
        self._set_axes()

        # get data
        n = self.config.get_n() if self._is_static() else self.config.get_n(self.time_point)[..., 0]
        self._assert_single_configuration(n)
        node_quantity, face_quantity, arrow_quantity = None, None, None
        if self.show_nodes and self.show_node_quantity:
            node_quantity = self._get_node_quantity()
            self._assert_single_configuration(node_quantity)
        if self.show_face_quantity:
            face_quantity = self._get_face_quantity()
            self._assert_single_configuration(face_quantity)
        if self.show_arrows:
            arrow_quantity = self._get_junction_quantity()
            self._assert_single_configuration(arrow_quantity)

        # plot data
        if self.show_grid:
            self._plot_grid()
        if self.show_face_quantity:
            self._plot_faces(face_quantity)
        if self.show_nodes:
            self._plot_nodes(node_quantity)
        if self.show_arrows:
            self._plot_arrows(arrow_quantity)
        if self.show_vortices:
            self._plot_vortices(n)

        # return handles
        if self.colorbar is not None:
            return self.fig, self.ax, self.colorbar
        else:
            return self.fig, self.ax


class CircuitMovie(Plot):

    def __init__(self, config: TimeEvolutionResult, problem_nr=0, time_points=None, show_vortices=True, vortex_diameter=0.25,
                 vortex_color=(0, 0, 0), anti_vortex_color=(0.8, 0.1, 0.2), vortex_alpha=1, show_grid=True,
                 grid_width=1, grid_color=(0.3, 0.5, 0.9), grid_alpha=0.5, show_colorbar=True, show_arrows=True,
                 arrow_quantity="I", arrow_width=0.005, arrow_scale=1, arrow_headwidth=3, arrow_headlength=5,
                 arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1, arrow_color=(0.2, 0.4, 0.7),
                 arrow_alpha=1, show_nodes=True, node_diameter=0.2, node_face_color=(1, 1, 1),
                 node_edge_color=(0, 0, 0), node_alpha=1, show_node_quantity=False, node_quantity="phase",
                 node_quantity_cmap=None, node_quantity_clim=(0, 1), node_quantity_alpha=1,
                 node_quantity_logarithmic_colors=False, show_face_quantity=False, face_quantity="n",
                 face_quantity_cmap=None, face_quantity_clim=(0, 1), face_quantity_alpha=1,
                 face_quantity_logarithmic_colors=False, figsize=None, animate_interval=5, title=""):

        super().__init__(config, time_points, show_vortices, vortex_diameter, vortex_color, anti_vortex_color, vortex_alpha,
                         show_grid, grid_width, grid_color, grid_alpha, show_colorbar, show_arrows, arrow_quantity,
                         arrow_width, arrow_scale, arrow_headwidth, arrow_headlength, arrow_headaxislength,
                         arrow_minshaft, arrow_minlength, arrow_color, arrow_alpha, show_nodes, node_diameter,
                         node_face_color, node_edge_color, node_alpha, show_node_quantity, node_quantity,
                         node_quantity_cmap, node_quantity_clim, node_quantity_alpha, node_quantity_logarithmic_colors,
                         show_face_quantity, face_quantity, face_quantity_cmap, face_quantity_clim, face_quantity_alpha,
                         face_quantity_logarithmic_colors, figsize, title)

        if self.time_point is None:
            self.time_point = np.ones(self.config.problem._Nt(), dtype=bool)
        if not (self.time_point.dtype in (bool, np.bool)):
            try:
                self.time_point = np.zeros(self.config.problem._Nt(), dtype=bool)
                self.time_point[time_points] = True
            except:
                raise ValueError("Invalid time_points; must be None, mask, slice or index array")
        self.problem_nr = problem_nr
        self.time_point &= self.config.problem.store_time_steps
        self.time_points_nz = np.flatnonzero(self.time_point)
        self.animate_interval = animate_interval
        self.faces_handle, self.nodes_handle, self.arrows_handle, self.vortices_handle = None, None, None, []
        self.item = None
        self.node_quantity_data = None
        self.face_quantity_data = None
        self.arrow_quantity_data = None
        self.n_data = None

    def show(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._set_axes()
        plt.title(self.title)

        self.n_data = self.config.get_n()[:, self.problem_nr, self.config._time_point_index(self.time_point)]
        if self.show_nodes and self.show_node_quantity:
            self.node_quantity_data = self._get_node_quantity()[:, self.problem_nr, :]
        if self.show_face_quantity:
            self.face_quantity_data = self._get_face_quantity()[:, self.problem_nr, :]
        if self.show_arrows:
            self.arrow_quantity_data = self._get_junction_quantity()[:, self.problem_nr, :]
        if self.show_grid:
            self._plot_grid()
        time_point_list = np.arange(self.n_data.shape[-1], dtype=int)
        self.ani = animation.FuncAnimation(self.fig, self._animate, time_point_list,
                                           init_func=self._init, interval=self.animate_interval, blit=True)
        if self.colorbar is not None:
            return self.ani, self.fig, self.ax, self.colorbar
        else:
            return self.ani, self.fig, self.ax

    def _get_time_point_mask(self, time_point):
        mask = np.zeros(self.config.problem._Nt(), dtype=bool)
        mask[time_point] = True
        return mask

    def _animate(self, i):
        if self.show_face_quantity:
            self.faces_handle = self._update_faces(i)
        if self.show_nodes:
            self.nodes_handle = self._update_nodes(i)
        if self.show_arrows:
            self.arrows_handle = self._update_arrows(i)
        if self.show_vortices:
            self.vortex_handles =self._plot_vortices(self.n_data[:, i])
        self.time_label.set_text(str(self.time_points_nz[i]))
        handles = [self.nodes_handle, self.arrows_handle, self.faces_handle, self.time_label] + self.vortex_handles
        return [h for h in handles if h is not None]

    def _init(self):
        if self.show_face_quantity:
            self.faces_handle = self._plot_faces(self.face_quantity_data[:, 0])
        if self.show_nodes and self.show_node_quantity:
            self.nodes_handle = self._plot_nodes(self.node_quantity_data[:, 0])
        if self.show_arrows:
            self.arrows_handle = self._plot_arrows(self.arrow_quantity_data[:, 0])
        if self.show_vortices:
            self.vortex_handles = self._plot_vortices(self.n_data[:, 0])
        handles = [self.faces_handle, self.nodes_handle, self.arrows_handle, self.time_label] + self.vortex_handles
        return [h for h in handles if h is not None]

    def _update_faces(self, i):
        face_quantity = self.face_quantity_data[:, i]
        self.faces_handle.set_array(face_quantity)
        return self.faces_handle

    def _update_nodes(self, i):
        if self.show_node_quantity:
            if self.node_quantity_data.ndim >= 2:
                node_quantity = self.node_quantity_data[:, i]
            else:
                node_quantity = self.node_quantity_data
            self.nodes_handle.set_array(node_quantity)
        return self.nodes_handle

    def _update_arrows(self, i):
        I = self.arrow_quantity_data[:, i] * self.arrow_scale
        x1, y1, x2, y2 = self.config.get_circuit().get_juncion_coordinates()
        xq, yq = x1 + 0.5 * (1 - I) * (x2 - x1), y1 + 0.5 * (1 - I) * (y2 - y1)
        U, V = I * (x2 - x1), I * (y2 - y1)
        self.arrows_handle.set_UVC(U, V)
        self.arrows_handle.X = xq
        self.arrows_handle.Y = yq
        self.arrows_handle.XY = np.concatenate((xq[:, None], yq[:, None]), axis=1)
        self.arrows_handle._offsets = self.arrows_handle.XY
        return self.arrows_handle

