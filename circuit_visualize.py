import time

import matplotlib
import numpy as np
from scipy.spatial import Voronoi

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import Normalize, LogNorm
import matplotlib.animation as animation
import matplotlib.cm as cm

from pyjjasim import Circuit

from pyjjasim.static_problem import StaticConfiguration
from pyjjasim.time_evolution import TimeEvolutionResult

__all__ = ["CircuitPlot", "CircuitMovie", "ConfigMovie", "ConfigMovie"]

# TODO:


class CircuitPlot:

    """Plots a circuit configuration.

    Allows one to show node quantities, arrow quantities, face quantities and vortices.
        - Node quantities are shown as colors on the nodes.
        - arrow quantities are displayed at the junctions and the length of the arrows is
          proportional to the quantity value.
        - face quantities are shown as colors on the faces.
        - vortices are displayed by symbols (concentric rings, vorticity equals nr of rings,
          color shows sign).

    Base class used by ConfigPlot, where one can specify what physical quantities to display
    in the circuit.

    """

    def __init__(self, circuit: Circuit, node_data=None,
                 arrow_data=None, face_data=None, vortex_data=None,
                 vortex_diameter=0.25, vortex_color=(0, 0, 0), anti_vortex_color=(0.8, 0.1, 0.2),
                 vortex_alpha=1, vortex_label="",
                 show_grid=True, grid_width=1, grid_color=(0.4, 0.5, 0.6), grid_alpha=0.5,
                 show_colorbar=True, show_legend=True, show_axes=True,
                 arrow_width=0.005, arrow_scale=1, arrow_headwidth=3, arrow_headlength=5,
                 arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1,
                 arrow_color=(0.15, 0.3, 0.8), arrow_alpha=1, arrow_label="",
                 show_nodes=True, node_diameter=0.25, node_face_color=(1,1,1),
                 node_edge_color=(0, 0, 0),
                 nodes_as_voronoi=False,
                 node_alpha=1, node_quantity_cmap=None,
                 node_quantity_clim=None, node_quantity_alpha=1,
                 node_quantity_logarithmic_colors=False, node_label="",
                 face_quantity_cmap=None, face_quantity_clim=None, face_quantity_alpha=1,
                 face_quantity_logarithmic_colors=False, face_label="",
                 figsize=None, title="", _vortex_range=None):

        """
        Constructor for Plot and handling plot options.

        Parameters
        ----------
        circuit: Circuit

        node_data=None: (Nn,) array or None
            Data associated with nodes in circuit. Can be visualized as colors.
            If None, node data is not visualized.
        arrow_data=None: (Nj,) array or None
            Data associated with junctions in circuit. Can be visualized as arrows.
            where the length corresponds with the magnitude of the data.
            If None, arrow data is not visualized.
        face_data=None: (Nf,) array or None
            Data associated with faces in circuit. Can be visualized as colors.
            If None, face data is not visualized.
        vortex_data=None: (Nf,) int array or None
            Integer data associated with faces in circuit. Can be visualized as
            circular symbols. The value of the data equals the number of concentric
            rings of the symbol. The color shows if it is + or -.
            If None, vortex data is not visualized.
        vortex_diameter=0.25: float
            diameter of vortex symbols
        vortex_color=(0, 0, 0): color
            color of vortex symbols.
        anti_vortex_color=(0.8, 0.1, 0.2): color
            color of anti-vortex symbols, whose data is negative.
        vortex_alpha=1: float
            transparancy of vortex symbols.
        vortex_label="": str
            label given to vortex data in legend.
        show_grid=True: bool
            display a grid at the edges of the graph.
        grid_width=1: float
            width of lines of grid.
        grid_color=(0.4, 0.5, 0.6): color
            color of grid.
        grid_alpha=0.5: float
            transparency of grid.
        show_colorbar=True: bool
            show colorbar mapping face and/or node data to colors.
        show_legend=True: bool
            show legend which includes colormaps, explanation of vortex sybols and
            an arrow scale.
        arrow_width=0.005: float
            width of arrows
        arrow_scale=1: float
            scale-factor for arrows. (length of arrow = arrow_scale * arrow_data)
        arrow_headwidth=3: float
            width of head of arrows
        arrow_headlength=5: float
            length of head of arrows
        arrow_headaxislength=4.5: float
            axislength of head of arrows
        arrow_minshaft=1: float
            arrow propertu
        arrow_minlength=1: float
            arrow property
        arrow_color=(0.15, 0.3, 0.8): color
            color of arrows.
        arrow_alpha=1: float
            transparency of arrows
        arrow_label="": str
            label given to arrow data in legend
        show_nodes=True: bool
            if True, nodes are displayed as circles
        node_diameter=0.25: float
            diameter of nodes.
        node_face_color=(1,1,1): color
            color of faces of nodes. Only used if there is no node data.
        node_edge_color=(0, 0, 0): color
            color of edge of nodes.  Only used if there is no node data.
        nodes_as_voronoi=False: bool
            if True, node data is visualized as colors of faces of a
            voronoi diagram based on node coordinates rather than color
            of circles at node coordinates.
        node_alpha=1: float
            transparency of nodes
        node_quantity_cmap=None: colormap or None
            colormap for node_data
        node_quantity_clim=None: (float, float) or None
            color limits for node_data
        node_quantity_alpha=1: float
            ...
        node_quantity_logarithmic_colors=False: bool
            if True, node_data color-scale is logarithmic
        node_label="": str
            label given to the node colormap
        face_quantity_cmap=None: colormap or None

        face_quantity_clim=None: (float, float) or None
        face_quantity_alpha=1: float
        face_quantity_logarithmic_colors=False: bool
        face_label="": str
        figsize=None: (float, float) or None
        title="": str

            coordinates of nodes of embedded graph
        node1, node2: (E,) int array in range(N)
            endpoint nodes of edges in embedded graph. Nodes are referred to by their index in
            the coordinate arrays.
        require_single_component=False:
            If True, an error is raised if the graph is not single-component
        require_planar_embedding=False:
            If True, an error is raised if the graph is not a planar embedding


        """
        self.circuit = circuit

        self.node_data = node_data
        self.arrow_data = arrow_data
        self.face_data = face_data

        self.vortex_data = np.array(vortex_data, dtype=int) if vortex_data is not None else vortex_data

        self.show_vortices = self.vortex_data is not None
        self.vortex_diameter = vortex_diameter
        self.vortex_color = vortex_color
        self.anti_vortex_color = anti_vortex_color
        self.vortex_alpha = vortex_alpha
        self.vortex_label = vortex_label
        self._assign_vortex_range(_vortex_range)

        self.show_grid = show_grid
        self.grid_width = grid_width
        self.grid_color = grid_color
        self.grid_alpha = grid_alpha
        self.show_colorbar = show_colorbar
        self.show_legend = show_legend
        self.show_axes = show_axes

        self.show_arrows = self.arrow_data is not None
        self.arrow_width = arrow_width
        self._assign_arrow_scale(arrow_scale)
        self.arrow_headwidth = arrow_headwidth
        self.arrow_headlength = arrow_headlength
        self.arrow_headaxislength = arrow_headaxislength
        self.arrow_minshaft = arrow_minshaft
        self.arrow_minlength = arrow_minlength
        self.arrow_color = arrow_color
        self.arrow_alpha = arrow_alpha
        self.arrow_label = arrow_label

        self.show_nodes = show_nodes
        self.node_diameter = node_diameter
        self.node_face_color = node_face_color
        self.node_edge_color = node_edge_color
        self.node_alpha = node_alpha
        self.show_node_quantity = (self.node_data is not None) and show_nodes
        self.node_quantity_cmap = node_quantity_cmap
        if self.node_quantity_cmap is None:
            self.node_quantity_cmap = "twilight"
        self.node_quantity_clim = node_quantity_clim
        self.node_quantity_alpha = node_quantity_alpha
        self.node_quantity_logarithmic_colors = node_quantity_logarithmic_colors
        self.node_label = node_label
        self.nodes_as_voronoi = nodes_as_voronoi

        self.show_face_quantity = self.face_data is not None
        self.face_quantity_cmap = face_quantity_cmap
        if self.face_quantity_cmap is None:
            self.face_quantity_cmap = "pink"
        self.face_quantity_clim = face_quantity_clim
        self.face_quantity_alpha = face_quantity_alpha
        self.face_quantity_logarithmic_colors = face_quantity_logarithmic_colors
        self.face_label = face_label

        self.figsize = figsize if figsize is not None else [6.4, 5]
        self.colorbar1 = None
        self.cb1_label = None
        self.colorbar2 = None
        self.cb2_label = None
        self.title = title
        self.fig = None
        self.ax = None
        self.ax_legend = None
        self.ax_cb1 = None
        self.ax_cb2 = None

        self.grid_handle = None
        self.node_handle = None
        self.face_handle = None
        self.arrow_handle = None
        self.vortex_handles = []

    def _prepare_legend(self, left, bottom, width, height):
        self.ax_legend = self.fig.add_axes([left, bottom, width, height])
        self.legend_stack = []
        self.legend_stack += [0.75] if self.show_arrows else [0.93]
        self.legend_stack += [0.75] if self.show_arrows else [0.93]
        s = self.show_vortices and len(self._vortex_range) > 0
        self.legend_stack += [self._vortex_legend_get_ymin(self.legend_stack[-1])[0]] if s else [self.legend_stack[-1]]
        s = max(0, self.legend_stack[-1] - 0.5)
        if self.show_node_quantity and self.show_face_quantity:
            self.ax_cb1 = self.ax_legend.inset_axes([0.01, 0.02 + s, 0.18, min(self.legend_stack[-1] - 0.1, 0.4)])
        else:   # at most one colorbar
            self.ax_cb1 = self.ax_legend.inset_axes([0.3, 0.02 + s, 0.18, min(self.legend_stack[-1] - 0.1, 0.4)])
        self.ax_cb2 = self.ax_legend.inset_axes([0.64, 0.02 + s, 0.18, min(self.legend_stack[-1] - 0.1, 0.4)])

    def make(self):
        self.fig = plt.figure(figsize=self.figsize)
        left, width, l_width = 0.1, 0.68, 0.17
        bottom, height = 0.1, 0.85
        spacing = 0.005

        if self.show_legend:
            self.ax = self.fig.add_axes([left, bottom, width, height])
        else:
            self.ax = self.fig.add_axes([left, bottom, width + l_width - 0.02, height])

        if self.show_legend:
            self._prepare_legend(left + width + spacing, bottom, l_width, height)

        self.ax.set_title(self.title)
        self._set_axes()
        if not self.show_axes:
            self.ax.set_axis_off()

        # plot data
        if self.show_grid:
            self._plot_grid()
        if self.show_face_quantity:
            self._plot_faces()
        if self.show_nodes:
            self._plot_nodes()
        if self.show_arrows:
            self._plot_arrows()
        if self.show_vortices:
            self._plot_vortices()
        if self.show_legend:
            if self.colorbar1 is None:
                self.ax_cb1.set_axis_off()
            if self.colorbar2 is None:
                self.ax_cb2.set_axis_off()
            self._make_legend()

        xmin, xmax, ymin, ymax = self._get_lims()
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        # return handles
        return self._return_figure_handles()

    def _return_figure_handles(self):
        handles = [self.fig, self.ax, self.colorbar1, self.colorbar2]
        return [h for h in handles if h is not None]

    def _assign_arrow_scale(self, arrow_scale):
        self.arrow_scale = arrow_scale
        if self.show_arrows and self.arrow_scale is None:
            self.arrow_scale = np.max(np.abs(self.arrow_data))
        self.arrow_scale = 1 if np.isclose(self.arrow_scale, 0) else self.arrow_scale

    def _assign_vortex_range(self, vortex_range):
        self._vortex_range = vortex_range
        if vortex_range is None and self.show_vortices:
            self._vortex_range = np.unique(self.vortex_data)
        if self._vortex_range is not None:
            self._vortex_range = self._vortex_range[self._vortex_range != 0]

    def _get_lims(self):
        x, y = self.circuit.get_node_coordinates()
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

    def _marker_scale_factor(self):
        return self._scale_factor(self.ax, True) * 72

    def _scale_factor(self, ax, on_x):
        x0, y0, width, height = ax.get_position().bounds
        if on_x:
            xlim = ax.get_xlim()
            return (width * self.figsize[0]) / (xlim[1] - xlim[0])
        else:
            ylim = ax.get_ylim()
            return (height * self.figsize[1]) / (ylim[1] - ylim[0])

    def _plot_grid(self):
        x1, y1, x2, y2 = self.circuit.get_juncion_coordinates()
        lines = [((x1[i], y1[i]), (x2[i], y2[i])) for i in range(len(x1))]
        lc = LineCollection(lines, colors=self.grid_color,  alpha=self.grid_alpha,linewidths=self.grid_width, zorder=0)
        self.ax.add_collection(lc)

    @staticmethod
    def _voronoi(ax, x, y, data, cnorm, cmap, alpha):
        points = np.stack((x.flatten(), y.flatten()), axis=0).T
        xm, xp = ax.get_xlim()
        ym, yp = ax.get_ylim()
        points = np.append(points, [[xm - 1, ym - 1], [xm - 1, yp + 1], [xp + 1, yp + 1], [xp + 1, ym - 1]], axis=0)
        vor = Voronoi(points)
        verts = [vor.vertices[region, :] for region in vor.regions]
        verts = [verts[i] for i in vor.point_region[:len(x)]]
        coll = PolyCollection(verts, array=data, edgecolors='none', cmap=cmap,
                              norm=cnorm, alpha=alpha, zorder=-1)
        return ax.add_collection(coll)

    def _cnorm(self, clim, data, logarithmic):
        if clim is None:
            clim = (np.min(data), np.max(data))
        return clim, Normalize(*clim) if not logarithmic else LogNorm(*clim)

    def _assign_colorbar(self, cmap, clim, label, cb_nr):
        if cb_nr == 0:
            self.colorbar1 = matplotlib.colorbar.Colorbar(ax=self.ax_cb1, mappable=cmap)
            self.cb1_label = label
            if np.allclose(clim, [-np.pi, np.pi]):
                self._set_ax_pi_ticks(self.colorbar1)
            else:
                self.ax_cb1.set_ylim(clim)
        else:
            self.colorbar2 = matplotlib.colorbar.Colorbar(ax=self.ax_cb2, mappable=cmap)
            self.cb2_label = label
            if np.allclose(clim, [-np.pi, np.pi]):
                self._set_ax_pi_ticks(self.colorbar2)
            else:
                self.ax_cb2.set_ylim(clim)

    def _plot_nodes(self):
        x, y = self.circuit.get_node_coordinates()

        marker_size = self.node_diameter * self._marker_scale_factor()
        if not self.show_node_quantity:
            node_handle = self.ax.plot(x, y, markeredgecolor=self.node_edge_color,
                                       markerfacecolor=self.node_face_color,
                                       linestyle="None",
                                       markersize=marker_size, marker="o",
                                       alpha=self.node_alpha, zorder=2)
            self.node_handle = node_handle[0]
        else:
            clim, cnorm = self._cnorm(self.node_quantity_clim, self.node_data, self.node_quantity_logarithmic_colors)
            if self.nodes_as_voronoi:
                self.node_handle = self._voronoi(self.ax, x, y, self.node_data, cnorm,
                                                 self.node_quantity_cmap, self.node_quantity_alpha)
            else:
                self.node_handle = self.ax.scatter(x.flatten(), y.flatten(), s=marker_size**2,
                                                   c=self.node_data, cmap=self.node_quantity_cmap,
                                                   edgecolors=None,
                                                   alpha=self.node_quantity_alpha, norm=cnorm)
            if self.show_colorbar and self.show_legend:
                cmap = cm.ScalarMappable(norm=cnorm, cmap=self.node_quantity_cmap)
                if self.colorbar1 is None:
                    self._assign_colorbar(cmap, clim, self.node_label, cb_nr=0)
                else:
                    self._assign_colorbar(cmap, clim, self.node_label, cb_nr=1)

    def _set_ax_pi_ticks(self, cb):
        cb.set_ticks([-np.pi, -0.5 * np.pi, 0, 0.5*np.pi, np.pi])
        cb.set_ticklabels(["$-\pi$", "$-\pi/2$", "$0$", "$\pi/2$", "$\pi$"])

    def _plot_arrows(self):
        I = self.arrow_data * self.arrow_scale
        x1, y1, x2, y2 = self.circuit.get_juncion_coordinates()
        xq = x1 + 0.5 * (1 - I) * (x2 - x1)
        yq = y1 + 0.5 * (1 - I) * (y2 - y1)
        dxq, dyq = I * (x2 - x1), I * (y2 - y1)
        self.arrow_handle = self.ax.quiver(xq, yq, dxq, dyq, edgecolor=self.arrow_color,
                                           facecolor=self.arrow_color, angles='xy', scale=1,
                                           scale_units='xy', width=self.arrow_width,
                                           headwidth=self.arrow_headwidth, headlength=self.arrow_headlength,
                                           headaxislength=self.arrow_headaxislength, minshaft=self.arrow_minshaft,
                                           minlength=self.arrow_minlength, alpha=self.arrow_alpha, zorder=3)

    def _plot_faces(self):
        face_nodes = self.circuit.get_faces()
        x, y = self.circuit.get_node_coordinates()
        verts = [np.stack((x[n], y[n]), axis=-1) for n in face_nodes]
        clim = self.face_quantity_clim
        if clim is None:
            clim = (np.min(self.face_data), np.max(self.face_data))
        cnorm = Normalize(*clim) if not self.face_quantity_logarithmic_colors else LogNorm(*clim)
        coll = PolyCollection(verts, array=self.face_data, edgecolors='none',
                              cmap=self.face_quantity_cmap,
                              norm=cnorm, alpha=self.face_quantity_alpha, zorder=-1)
        self.face_handle = self.ax.add_collection(coll)
        if self.show_colorbar and self.show_legend:
            cmap = cm.ScalarMappable(norm=cnorm, cmap=self.face_quantity_cmap)
            self._assign_colorbar(cmap, clim, self.face_label, cb_nr=0)

    def _plot_vortices(self):
        n = self.vortex_data
        marker_size = self.vortex_diameter * self._marker_scale_factor()
        xc, yc = self.circuit.get_face_centroids()
        ns = self._vortex_range
        vort_handles = []
        for ni in ns:
            n_mask = n == ni
            xni, yni = xc[n_mask], yc[n_mask]
            p = self._draw_vortex(self.ax, xni, yni, ni, self.vortex_color,
                                  self.anti_vortex_color, marker_size, self.vortex_alpha)
            vort_handles += p
        self.vortex_handles = vort_handles

    def _update_nodes(self, new_node_data):
        if new_node_data is not None and self.show_node_quantity:
            self.node_handle.set_array(new_node_data)

    def _update_arrows(self, new_arrow_data):
        if new_arrow_data is not None and self.show_arrows:
            I = new_arrow_data * self.arrow_scale
            x1, y1, x2, y2 = self.circuit.get_juncion_coordinates()
            xq, yq = x1 + 0.5 * (1 - I) * (x2 - x1), y1 + 0.5 * (1 - I) * (y2 - y1)
            U, V = I * (x2 - x1), I * (y2 - y1)
            self.arrow_handle.set_UVC(U, V)
            self.arrow_handle.X = xq
            self.arrow_handle.Y = yq
            self.arrow_handle.XY = np.concatenate((xq[:, None], yq[:, None]), axis=1)
            self.arrow_handle._offsets = self.arrow_handle.XY

    def _update_faces(self, new_face_data):
        if new_face_data is not None and self.show_face_quantity:
            self.face_handle.set_array(new_face_data)

    def _update_vortices(self, new_vortex_data):
        if new_vortex_data is not None and self.show_vortices:
            n = new_vortex_data
            xc, yc = self.circuit.get_face_centroids()
            ns = self._vortex_range
            index = 0
            for i, ni in enumerate(ns):
                na = np.abs(ni)
                for handle in self.vortex_handles[index:(index + 2 * na)]:
                    handle.set_xdata(xc[n == ni])
                    handle.set_ydata(yc[n == ni])
                index += 2 * na - 1

    def _make_legend(self):
        self.ax_legend.set_xlim([0, 1])
        self.ax_legend.set_ylim([0, 1])
        self.ax_legend.set_axis_off()

        # arrow legend
        self.ax_legend.text(0.5, 0.975, "legend", ha="center", va="center", fontsize=13)
        if self.show_arrows:
            self._arrow_legend(0.75, 0.93)

        # vortex legend
        if self.show_vortices and len(self._vortex_range) > 0:
            self._vortex_legend(self.legend_stack[0])

        # node/face legend
        if self.cb1_label is not None:
            if self.cb2_label is not None:
                self.ax_legend.text(0.11, self.legend_stack[-1] - 0.05, self.cb1_label, ha="center", va="center")
            else:
                self.ax_legend.text(0.4, self.legend_stack[-1] - 0.05, self.cb1_label, ha="center", va="center")
        if self.cb2_label is not None:
            self.ax_legend.text(0.75, self.legend_stack[-1] - 0.05, self.cb2_label, ha="center", va="center")

    @staticmethod
    def _nearest_anchor(x):
        b = np.log(x)/np.log(10)
        decade = np.floor(b)
        B = b - decade
        s = np.array([0, np.log(2)/np.log(10), np.log(5)/np.log(10), 1])
        sub = np.argmin(np.abs(B - s))
        if sub == 0:
            return 10 ** decade
        if sub == 1:
            return 2 * 10 ** decade
        if sub == 2:
            return 5 * 10 ** decade
        if sub == 3:
            return 10 ** (decade + 1)

    def _legend_arrow_scale(self):
        _, _, width, _ = self.ax.get_position().bounds
        _, _, width_L, _ = self.ax_legend.get_position().bounds
        xlim = self.ax.get_xlim()
        xlim_L = self.ax_legend.get_xlim()
        v = np.median(self.circuit._junction_lengths())
        dx = xlim[1] - xlim[0]
        dx_L = xlim_L[1] - xlim_L[0]
        return v * (width / dx) / (width_L / dx_L), v / self.arrow_scale

    def _arrow_legend(self, y_min, y_max):
        L, label = self._legend_arrow_scale()
        label *= (0.51 / L)
        L = 0.51
        new_label = self._nearest_anchor(label)
        L *= new_label / label
        ax =self.ax_legend
        y = 0.6 * y_min + 0.4 * y_max
        xc = 0.58
        hL = L/2
        ax.text(0.12, y, self.arrow_label, ha="center", va="center")
        ax.quiver([xc-hL], [y], [L], [0], edgecolor=self.arrow_color, facecolor=self.arrow_color,
                  scale_units="xy", angles="xy", scale=1, width=0.02)
        dy = 0.01
        Y = 0.4 * y_min + 0.6 * y_max
        ax.plot([xc-hL, xc-hL, xc-hL, xc+hL, xc+hL, xc+hL], [Y-dy, Y+dy, Y, Y, Y+dy, Y-dy], color=[0, 0, 0])
        yy = 0.2 * y_min + 0.8 * y_max
        ax.text(xc, yy, f"{new_label}", ha="center", va="center")

    def _vortex_legend_get_ymin(self, y_max):
        n_range= self._vortex_range
        y_min = -np.inf
        scale_factor_x = self._scale_factor(self.ax_legend, True)
        scale_factor_y = self._scale_factor(self.ax_legend, False)
        d = 0.3
        while y_min < 0.25:
            d *= 0.8
            Dy = 1.1 * d * (len(n_range) + 1) / (scale_factor_y / scale_factor_x)
            y_min = y_max - Dy
        return y_min, d

    def _vortex_legend(self, y_max):
        ax = self.ax_legend
        n_range= self._vortex_range
        v_color, av_color = self.vortex_color, self.anti_vortex_color
        label = self.vortex_label
        scale_factor_x = self._scale_factor(ax, True)
        y_min, d = self._vortex_legend_get_ymin(y_max)
        Y = np.linspace(y_max, y_min, 2 * len(n_range) + 2)
        y = Y[2:-1:2]
        ax.text(0.12, Y[1], label, ha="center", va="bottom")
        ax.text(0.6, Y[1], "symbol", ha="center", va="bottom")
        for i, n in enumerate(n_range):
            CircuitPlot._draw_vortex(ax, 0.6, y[i], n, v_color, av_color, scale_factor_x * d * 72, 1)
            ax.text(0.12, y[i], f"{n}", va="center", ha="center")
        plt.plot([0.02, 0.99, 0.99, 0.02, 0.02], [y_min, y_min, y_max+0.016, y_max+0.016, y_min], color=[0, 0, 0])
        return y_min

    @staticmethod
    def _draw_vortex(ax, x, y, n, v_color, av_color, diameter, alpha):
        color = v_color if n > 0 else av_color
        na = np.abs(n)
        handles = []
        for k in reversed(range(na)):
            frac = (2 * k + 1) / (2 * na - 1)
            p = ax.plot(x, y, markerfacecolor=color, markeredgecolor=color, marker="o",
                        linestyle="", markersize=frac * diameter, alpha=alpha, zorder=4)
            handles += p
            if k > 0:
                frac = (2 * k) / (2 * na - 1)
                p = ax.plot(x, y, markerfacecolor=[1, 1, 1], markeredgecolor=[1, 1, 1], marker="o",
                            linestyle="", markersize=frac * diameter, alpha=alpha, zorder=4)
                handles += p
        return handles


class CircuitMovie(CircuitPlot):
    """Animation of circuit where the quantities evolve over time.

    Allows one to show node quantities, arrow quantities, face quantities and vortices.
        - Node quantities are shown as colors on the nodes.
        - arrow quantities are displayed at the junctions and the length of the arrows is
          proportional to the quantity value.
        - face quantities are shown as colors on the faces.
        - vortices are displayed by symbols (concentric rings, vorticity equals nr of rings,
          color shows sign).

    Base class used by ConfigMovie, where one can specify what physical quantities to display
    in the circuit.

    node_data: (Nn, time_points) or (Nn,) or None
    """

    def __init__(self, circuit: Circuit, animate_interval=10,
                 node_data=None, arrow_data=None, face_data=None, vortex_data=None,
                 vortex_diameter=0.25, vortex_color=(0, 0, 0), anti_vortex_color=(0.8, 0.1, 0.2),
                 vortex_alpha=1, vortex_label="", _vortex_range=None,
                 show_grid=True, grid_width=1, grid_color=(0.4, 0.5, 0.6), grid_alpha=0.5,
                 show_colorbar=True, show_legend=True, show_axes=True,
                 arrow_width=0.005, arrow_scale=1, arrow_headwidth=3, arrow_headlength=5,
                 arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1,
                 arrow_color=(0.15, 0.3, 0.8), arrow_alpha=1, arrow_label="",
                 show_nodes=True, node_diameter=0.25, node_face_color=(1,1,1),
                 node_edge_color=(0, 0, 0),
                 nodes_as_voronoi=False,
                 node_alpha=1, node_quantity_cmap=None,
                 node_quantity_clim=None, node_quantity_alpha=1,
                 node_quantity_logarithmic_colors=False, node_label="",
                 face_quantity_cmap=None, face_quantity_clim=None, face_quantity_alpha=1,
                 face_quantity_logarithmic_colors=False, face_label="",
                 figsize=None, title=""):

        def handle_data(data, N):
            if data is None:
                return None, 1
            data = np.array(data)
            if data.ndim == 1:
                data = data[:, None]
            if data.ndim != 2 or data.shape[0] != N:
                raise ValueError(f"data must None, of shape ({N},), ({N}, 1) or ({N}, Nt)")
            Nt = data.shape[1]
            return data, Nt

        self.all_node_data, Ntn = handle_data(node_data, circuit.node_count())
        self.all_arrow_data, Nta = handle_data(arrow_data, circuit.junction_count())
        self.all_face_data, Ntf = handle_data(face_data, circuit.face_count())
        self.all_vortex_data, Ntv = handle_data(vortex_data, circuit.face_count())
        Nts = np.array([Ntn, Nta, Ntf, Ntv], dtype=int)
        self.Nt = np.max(Nts)
        if not np.all((Nts == 1) | (Nts == self.Nt)):
            raise ValueError("Not all data has same number of timesteps")

        if self.all_node_data is not None:
            if node_quantity_clim is None:
                node_quantity_clim = (np.min(self.all_node_data), np.max(self.all_node_data))
        if self.all_face_data is not None:
            if face_quantity_clim is None:
                face_quantity_clim = (np.min(self.all_face_data), np.max(self.all_face_data))

        super().__init__(circuit,  node_data=self.all_node_data[:, 0] if self.all_node_data is not None else None,
                         arrow_data=self.all_arrow_data[:, 0] if self.all_arrow_data is not None else None,
                         face_data=self.all_face_data[:, 0] if self.all_face_data is not None else None,
                         vortex_data=self.all_vortex_data[:, 0] if self.all_vortex_data is not None else None,
                         vortex_label=vortex_label, node_label=node_label, arrow_label=arrow_label, face_label=face_label,
                         vortex_diameter=vortex_diameter, vortex_color=vortex_color,
                         anti_vortex_color=anti_vortex_color,
                         vortex_alpha=vortex_alpha,
                         show_grid=show_grid, grid_width=grid_width, grid_color=grid_color,
                         grid_alpha=grid_alpha, show_colorbar=show_colorbar, show_legend=show_legend,
                         show_axes=show_axes, arrow_width=arrow_width, arrow_scale=arrow_scale,
                         arrow_headwidth=arrow_headwidth, arrow_headlength=arrow_headlength,
                         arrow_headaxislength=arrow_headaxislength, arrow_minshaft=arrow_minshaft, arrow_minlength=arrow_minlength,
                         arrow_color=arrow_color, arrow_alpha=arrow_alpha, show_nodes=show_nodes,
                         node_diameter=node_diameter, node_face_color=node_face_color, node_edge_color=node_edge_color,
                         node_alpha=node_alpha, nodes_as_voronoi=nodes_as_voronoi,
                         node_quantity_cmap=node_quantity_cmap, node_quantity_clim=node_quantity_clim,
                         node_quantity_alpha=node_quantity_alpha, node_quantity_logarithmic_colors=node_quantity_logarithmic_colors,
                         face_quantity_cmap=face_quantity_cmap,
                         face_quantity_clim=face_quantity_clim, face_quantity_alpha=face_quantity_alpha,
                         face_quantity_logarithmic_colors=face_quantity_logarithmic_colors, figsize=figsize, title=title)

        self.animate_interval = animate_interval

    def _init(self):
        handles = [self.face_handle, self.node_handle, self.arrow_handle] + self.vortex_handles
        return [h for h in handles if h is not None]

    def _animate(self, i):
        if self.show_face_quantity:
            if self.all_face_data.shape[1] > 1:
                self._update_faces(self.all_face_data[:, i])
        if self.show_node_quantity:
            if self.all_node_data.shape[1] > 1:
                self._update_nodes(self.all_node_data[:, i])
        if self.show_arrows:
            if self.all_arrow_data.shape[1] > 1:
                self._update_arrows(self.all_arrow_data[:, i])
        if self.show_vortices:
            if self.all_vortex_data.shape[1] > 1:
                self._update_vortices(self.all_vortex_data[:, i])
        handles = [self.node_handle, self.arrow_handle, self.face_handle] + self.vortex_handles
        return [h for h in handles if h is not None]

    def make(self):
        handles = super().make()
        self.ani = animation.FuncAnimation(self.fig, self._animate, np.arange(self.Nt, dtype=int),
                                           init_func=self._init, interval=self.animate_interval,
                                           blit=True)
        return [self.ani] + handles

    def _assign_arrow_scale(self, arrow_scale):
        self.arrow_scale = arrow_scale
        if self.show_arrows and self.arrow_scale is None:
            self.arrow_scale = np.max(np.abs(self.all_arrow_data))
        self.arrow_scale = 1 if np.isclose(self.arrow_scale, 0) else self.arrow_scale

    def _assign_vortex_range(self, vortex_range):
        self._vortex_range = vortex_range
        if vortex_range is None and self.show_vortices:
            self._vortex_range = np.unique(self.all_vortex_data)
        if self._vortex_range is not None:
            self._vortex_range = self._vortex_range[self._vortex_range != 0]


class ConfigPlot(CircuitPlot):

    def __init__(self, config: StaticConfiguration, node_quantity=None,
                 junction_quantity="I", face_quantity=None, vortex_quantity="n",
                 vortex_diameter=0.25, vortex_color=(0, 0, 0), anti_vortex_color=(0.8, 0.1, 0.2),
                 vortex_alpha=1, _vortex_range=None,
                 show_grid=True, grid_width=1, grid_color=(0.4, 0.5, 0.6), grid_alpha=0.5,
                 show_colorbar=True, show_legend=True, show_axes=True,
                 arrow_width=0.005, arrow_scale=1, arrow_headwidth=3, arrow_headlength=5,
                 arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1,
                 arrow_color=(0.15, 0.3, 0.8), arrow_alpha=1,
                 show_nodes=True, node_diameter=0.25, node_face_color=(1,1,1),
                 node_edge_color=(0, 0, 0), nodes_as_voronoi=False, node_alpha=1,
                 node_quantity_cmap=None,
                 node_quantity_clim=None, node_quantity_alpha=1,
                 node_quantity_logarithmic_colors=False,
                 face_quantity_cmap=None, face_quantity_clim=None, face_quantity_alpha=1,
                 face_quantity_logarithmic_colors=False,
                 figsize=None, title=""):

        self.config = config
        self.node_quantity = node_quantity if node_quantity is not None else ""
        self.arrow_quantity = junction_quantity if junction_quantity is not None else ""
        self.face_quantity = face_quantity if face_quantity is not None else ""
        self.vortex_quantity = vortex_quantity if vortex_quantity is not None else ""

        node_data, node_label = self._get_node_quantity()
        arrow_data, arrow_label = self._get_junction_quantity()
        face_data, face_label = self._get_face_quantity()
        vortex_data, vortex_label = self._get_vortex_quantity()

        if node_label == "phi":
            if node_quantity_clim is None:
                node_quantity_clim = (-np.pi, np.pi)

        if arrow_label == "th":
            if arrow_scale is None:
                arrow_scale = 1/np.pi

        super().__init__(config.get_circuit(),  node_data = node_data, arrow_data=arrow_data, face_data=face_data, vortex_data=vortex_data,
                         vortex_label=vortex_label, node_label=node_label, arrow_label=arrow_label, face_label=face_label,
                         vortex_diameter=vortex_diameter, vortex_color=vortex_color,
                         anti_vortex_color=anti_vortex_color,
                         vortex_alpha=vortex_alpha, show_grid=show_grid, grid_width=grid_width, grid_color=grid_color,
                         grid_alpha=grid_alpha, show_colorbar=show_colorbar, show_legend=show_legend,
                         show_axes=show_axes, arrow_width=arrow_width, arrow_scale=arrow_scale,
                         arrow_headwidth=arrow_headwidth, arrow_headlength=arrow_headlength,
                         arrow_headaxislength=arrow_headaxislength, arrow_minshaft=arrow_minshaft, arrow_minlength=arrow_minlength,
                         arrow_color=arrow_color, arrow_alpha=arrow_alpha, show_nodes=show_nodes,
                         node_diameter=node_diameter, node_face_color=node_face_color, node_edge_color=node_edge_color,
                         node_alpha=node_alpha, nodes_as_voronoi=nodes_as_voronoi,
                         node_quantity_cmap=node_quantity_cmap, node_quantity_clim=node_quantity_clim,
                         node_quantity_alpha=node_quantity_alpha, node_quantity_logarithmic_colors=node_quantity_logarithmic_colors,
                         face_quantity_cmap=face_quantity_cmap,
                         face_quantity_clim=face_quantity_clim, face_quantity_alpha=face_quantity_alpha,
                         face_quantity_logarithmic_colors=face_quantity_logarithmic_colors, figsize=figsize, title=title)


    _node_quantities = {
        "": -1,
        "phi": 0, "phase": 0, "phases": 0,
        "I_s": 1, "Is": 1, "current_sources": 1,
    }

    _junction_quantities = {
        "": -1,
        "th": 0, "theta": 0, "phase_difference": 0, "gauge_invariant_phase_difference": 0,
        "I": 1, "current": 1, "Isup": 1, "I_sup": 1, "Isuper": 1, "I_super": 1,
        "supercurrent": 1, "super_current": 1,
        "I_s": 2, "Is": 2, "current_sources": 2,
        "EJ": 3, "josephson_energy": 3,
        "EM": 4, "magnetic_energy": 4,
        "Etot": 5, "E_tot": 5, "ETot": 5, "total_energy": 5, "energy": 5,
    }

    _face_quantities = {
        "": -1,
        "Phi": 0, "flux": 0, "magnetic_flux": 0,
        "n": 1, "vortices": 1, "vortex_configuration": 1,
        "face_current": 2, "J": 2,
    }

    _vortex_quantities = {
        "": -1,
        "n": 0, "vortices": 0, "vortex_configuration": 0,
    }

    def _get_node_quantity(self):
        """
        Get node quantity (either "phi": gauge dependent phases or  "U": potential)
        """
        if isinstance(self.node_quantity, np.ndarray):
            return self.node_quantity.flatten(), "custom"
        quantity = self._node_quantities[self.node_quantity]
        if quantity == -1:   # none
            return None, None
        if quantity == 0:  # phi
            out = self.config.get_phi()
            out = out.copy()
            out -= np.round(out / (np.pi * 2.0)).astype(out.dtype) * np.pi * 2.0
            return out, "phi"
        if quantity == 1:   # Is
            return None, None

    def _get_junction_quantity(self):
        if isinstance(self.arrow_quantity, np.ndarray):
            return self.arrow_quantity.flatten(), "custom"
        quantity = self._junction_quantities[self.arrow_quantity]
        if quantity == -1:   # none
            return None, None
        if quantity == 0:  # theta
            out = self.config.get_theta()
            out = out.copy()
            out -= np.round(out / (np.pi * 2.0)).astype(out.dtype) * np.pi * 2.0
            return out, "th"
        if quantity == 1:  # I
            return self.config.get_I(), "I"
        if quantity == 2:  # Is
            return self.config.problem._Is(), "Is"
        if quantity == 3:  # EJ
            return self.config.get_EJ(), "EJ"
        if quantity == 4:  # EM
            return self.config.get_EM(), "EM"
        if quantity == 5:  # Etot
            return self.config.get_Etot(), "Etot"

    def _get_face_quantity(self):
        if isinstance(self.face_quantity, np.ndarray):
            return self.face_quantity.flatten(), "custom"
        quantity = self._face_quantities[self.face_quantity]
        if quantity == -1:   # none
            return None, None
        if quantity == 0:  # Phi
            return self.config.get_flux(), "flux"
        if quantity == 1:  # n
            return self.config.get_n(), "n"
        if quantity == 2:  # J
            return self.config.get_J(), "J"

    def _get_vortex_quantity(self):
        if isinstance(self.vortex_quantity, np.ndarray):
            return self.vortex_quantity.flatten().astype(int), "custom"
        quantity = self._vortex_quantities[self.vortex_quantity]
        if quantity == -1:   # none
            return None, None
        if quantity == 0:  # n
            return self.config.get_n(), "n"

    @staticmethod
    def _assert_single_configuration(data):
        if data.ndim >= 2:
            raise ValueError("must select single configuration")


class ConfigMovie(CircuitMovie):

    def __init__(self, config: TimeEvolutionResult, problem_nr=0, time_points=None, node_quantity=None,
                 junction_quantity="I", face_quantity=None, vortex_quantity="n",
                 vortex_diameter=0.25, vortex_color=(0, 0, 0), anti_vortex_color=(0.8, 0.1, 0.2),
                 vortex_alpha=1, _vortex_range=None,
                 show_grid=True, grid_width=1, grid_color=(0.4, 0.5, 0.6), grid_alpha=0.5,
                 show_colorbar=True, show_legend=True, show_axes=True,
                 arrow_width=0.005, arrow_scale=1, arrow_headwidth=3, arrow_headlength=5,
                 arrow_headaxislength=4.5, arrow_minshaft=1, arrow_minlength=1,
                 arrow_color=(0.15, 0.3, 0.8), arrow_alpha=1,
                 show_nodes=True, node_diameter=0.25, node_face_color=(1,1,1),
                 node_edge_color=(0, 0, 0),  nodes_as_voronoi=False,
                 node_alpha=1, node_quantity_cmap=None,
                 node_quantity_clim=None, node_quantity_alpha=1,
                 node_quantity_logarithmic_colors=False,
                 face_quantity_cmap=None, face_quantity_clim=None, face_quantity_alpha=1,
                 face_quantity_logarithmic_colors=False,
                 figsize=None, title="", animate_interval=10):

        self.config = config
        self.problem_nr = problem_nr
        self.time_points = time_points
        if self.time_points is None:
            self.time_points = np.ones(self.config.problem._Nt(), dtype=bool)
        if not (self.time_points.dtype in (bool, np.bool)):
            try:
                self.time_points = np.zeros(self.config.problem._Nt(), dtype=bool)
                self.time_points[time_points] = True
            except:
                raise ValueError("Invalid time_points; must be None, mask, slice or index array")
        self.time_points &= self.config.problem.store_time_steps

        def _get_quantity(quantity, getter_func, problem_nr):
            """
            quantity -> "", "I", ndarray
            """
            if quantity == "" or quantity is None:
                return "none", None
            elif isinstance(quantity, str):
                all_data, quantity = getter_func(quantity)
            else:
                all_data, quantity = quantity, "custom"
            return quantity, all_data[:, problem_nr, :]

        self.node_quantity, node_data = _get_quantity(node_quantity, self._get_node_quantity, problem_nr)
        self.arrow_quantity, arrow_data = _get_quantity(junction_quantity, self._get_junction_quantity, problem_nr)
        self.face_quantity, face_data = _get_quantity(face_quantity, self._get_face_quantity, problem_nr)
        self.vortex_quantity, vortex_data = _get_quantity(vortex_quantity, self._get_vortex_quantity, problem_nr)

        if self.node_quantity == "phi":
            if node_quantity_clim is None:
                node_quantity_clim = (-np.pi, np.pi)

        if self.arrow_quantity == "th":
            if arrow_scale is None:
                arrow_scale = 1/np.pi

        super().__init__(config.get_circuit(),  node_data=node_data, arrow_data=arrow_data,
                         face_data=face_data, vortex_data=vortex_data,
                         vortex_label=self.vortex_quantity, node_label=self.node_quantity,
                         arrow_label=self.arrow_quantity, face_label=self.face_quantity,
                         vortex_diameter=vortex_diameter, vortex_color=vortex_color,
                         anti_vortex_color=anti_vortex_color,
                         vortex_alpha=vortex_alpha, show_grid=show_grid, grid_width=grid_width,
                         grid_color=grid_color,
                         grid_alpha=grid_alpha, show_colorbar=show_colorbar, show_legend=show_legend,
                         show_axes=show_axes,
                         arrow_width=arrow_width, arrow_scale=arrow_scale,
                         arrow_headwidth=arrow_headwidth, arrow_headlength=arrow_headlength,
                         arrow_headaxislength=arrow_headaxislength, arrow_minshaft=arrow_minshaft,
                         arrow_minlength=arrow_minlength,
                         arrow_color=arrow_color, arrow_alpha=arrow_alpha, show_nodes=show_nodes,
                         node_diameter=node_diameter, node_face_color=node_face_color,
                         node_edge_color=node_edge_color, nodes_as_voronoi=nodes_as_voronoi,
                         node_alpha=node_alpha,
                         node_quantity_cmap=node_quantity_cmap, node_quantity_clim=node_quantity_clim,
                         node_quantity_alpha=node_quantity_alpha,
                         node_quantity_logarithmic_colors=node_quantity_logarithmic_colors,
                         face_quantity_cmap=face_quantity_cmap,
                         face_quantity_clim=face_quantity_clim, face_quantity_alpha=face_quantity_alpha,
                         face_quantity_logarithmic_colors=face_quantity_logarithmic_colors,
                         figsize=figsize, title=title, animate_interval=animate_interval)


    _node_quantities = {
        "phi": 0, "phase": 0, "phases": 0,
        "I_s": 1, "Is": 1, "current_sources": 1,
        "U": 2, "potential": 2,
    }

    _junction_quantities = {
        "th": 0, "theta": 0, "phase_difference": 0, "gauge_invariant_phase_difference": 0,
        "I": 1, "current": 1,
        "V": 2, "voltage": 2,
        "Isup": 3, "I_sup": 3, "Isuper": 3, "I_super": 3, "supercurrent": 3, "super_current": 3,
        "I_s": 4, "Is": 4, "current_sources": 4,
        "EJ": 5, "josephson_energy": 5,
        "EM": 6, "magnetic_energy": 6,
        "EC": 7, "capacitive_energy": 7, "capacitance_energy": 7,
        "Etot": 8, "E_tot": 8, "ETot": 8, "total_energy": 8, "energy": 8,
    }

    _face_quantities = {
        "Phi": 0, "flux": 0, "magnetic_flux": 0,
        "n": 1, "vortices": 1, "vortex_configuration": 1,
        "face_current": 2, "J": 2,
    }

    _vortex_quantities = {
        "n": 0, "vortices": 0, "vortex_configuration": 0,
    }

    def _get_node_quantity(self, node_quantity):
        """
        Get node quantity (either "phi": gauge dependent phases or  "U": potential)
        """
        quantity = self._node_quantities[node_quantity]
        if quantity == 0:  # phi
            out = self.config.get_phi(self.time_points)
            out = out.copy()
            out -= np.round(out / (np.pi * 2.0)).astype(out.dtype) * np.pi * 2.0
            return out, "phi"
        if quantity == 1:   # Is
            return self.config.problem.get_node_current_sources(self.time_points), "Is_node"
        if quantity == 2:   # U
            return self.config.get_U(self.time_points), "U"

    def _get_junction_quantity(self, junction_quantity):
        quantity = self._junction_quantities[junction_quantity]
        if quantity == 0:  # theta
            out = self.config.get_theta(self.time_points)
            out = out.copy()
            out -= np.round(out / (np.pi * 2.0)).astype(out.dtype) * np.pi * 2.0
            return out, "th"
        if quantity == 1:  # I
            return self.config.get_I(self.time_points), "I"
        if quantity == 2:  # V
            return self.config.get_V(self.time_points), "V"
        if quantity == 3:  # supercurrent
            return self.config.get_Isup(self.time_points), "Isup"
        if quantity == 4:  # Is
            return self.config.problem._Is(self.time_points), "Is"
        if quantity == 5:  # EJ
            return self.config.get_EJ(self.time_points), "EJ"
        if quantity == 6:  # EM
            return self.config.get_EM(self.time_points), "EM"
        if quantity == 7:  # EC
            return self.config.get_EC(self.time_points), "EC"
        if quantity == 8:  # Etot
            return self.config.get_Etot(self.time_points), "Etot"

    def _get_face_quantity(self, face_quantity):
        quantity = self._face_quantities[face_quantity]
        if quantity == 0:  # Phi
            return self.config.get_flux(self.time_points), "flux"
        if quantity == 1:  # n
            return self.config.get_n(self.time_points), "n"
        if quantity == 2:  # J
            return self.config.get_J(self.time_points), "J"

    def _get_vortex_quantity(self, vortex_quantity):
        quantity = self._vortex_quantities[vortex_quantity]
        if quantity == 0:  # n
            return self.config.get_n(self.time_points), "n"

