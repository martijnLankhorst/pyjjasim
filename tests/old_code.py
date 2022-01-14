def dual_graph(self, bounding_box=None):
    def get_e_edge(A, idx_nb, idx_b):
        mat = A[idx_nb, :].multiply(A[idx_b, :]).tocoo()
        _, index = np.unique(mat.row, return_index=True)
        return mat.col[index]

    def get_new_coordinates(n1, n2, x, y, xc, yc, bounding_box):
        L = len(xc)
        e = np.ones(L)
        X = 0.5 * (x[n1] + x[n2])
        Y = 0.5 * (y[n1] + y[n2])
        bx1, bx2 = bounding_box[0], bounding_box[0] + bounding_box[2]
        by1, by2 = bounding_box[1], bounding_box[1] + bounding_box[3]
        l = np.stack([(bx1 - xc) / (X - xc), (bx2 - xc) / (X - xc),
                      (by1 - yc) / (Y - yc), (by2 - yc) / (Y - yc)], axis=0)
        Xo = np.stack([bx1 * e, bx2 * e, xc + l[2, :] * (X - xc), xc + l[3, :] * (X - xc)], axis=0)
        Yo = np.stack([yc + l[0, :] * (Y - yc), yc + l[1, :] * (Y - yc), by1 * e, by2 * e], axis=0)
        idx = np.argmax(1 / np.array(l), axis=0)
        return Xo[(idx, np.arange(L))], Yo[(idx, np.arange(L))]

    x, y = self.coo()
    X, Y = self.get_l_cycle_centroids()
    n1, n2 = self.node1, self.node2
    if self.get_num_components() != 1:
        raise ValueError("number of components must be 1")

    k = 0.01
    if bounding_box is None:
        xl, xh, yl, yh = np.min(X), np.max(X), np.min(Y), np.max(Y)
        bounding_box = [(1 + k) * xl - k * xh, (1 + k) * yl - k * yh, (1 + 2 * k) * (xh - xl), (1 + 2 * k) * (yh - yl)]

    Al = self.l_cycle_matrix(_permute=False)

    adjl = (Al @ Al.T).tocoo()
    mask = adjl.row < adjl.col

    N1, N2 = adjl.row[mask], adjl.col[mask]  # in range(Nfl)

    b_mask = ~self._non_boundary_mask()

    edge_cases = b_mask[N1] | b_mask[N2]
    b_is_N1 = b_mask[N1][edge_cases]
    N1e = N1[edge_cases]
    N2e = N2[edge_cases]
    Ne_nb = np.where(b_is_N1, N2e, N1e)
    Ne_b = np.where(b_is_N1, N1e, N2e)
    e_edge = get_e_edge(Al, Ne_nb, Ne_b)
    xn, yn = get_new_coordinates(n1[e_edge], n2[e_edge], x, y, X[Ne_nb], Y[Ne_nb], bounding_box)

    new_node_idx = len(X) + np.arange(len(xn))

    N1e[b_is_N1] = new_node_idx[b_is_N1]
    N2e[~b_is_N1] = new_node_idx[~b_is_N1]
    N1[edge_cases] = N1e
    N2[edge_cases] = N2e

    X = np.append(X, xn)
    Y = np.append(Y, yn)

    return EmbeddedGraph(X, Y, N1, N2)

    def extended_graph(self, bounding_box=None, box_extend_factor=1.2):
        self._assign_faces()
        self._assign_boundary_faces()
        x, y = self.coo()
        n1, n2 = self.node1, self.node2
        if self.get_num_components() != 1:
            raise ValueError("number of components must be 1")
        k = box_extend_factor - 1
        if bounding_box is None:
            xl, xh, yl, yh = np.min(x), np.max(x), np.min(y), np.max(y)
            bounding_box = [(1+k) * xl - k * xh, (1+k) * yl - k * yh, (1+2*k) * (xh - xl), (1+2*k) *(yh - yl)]
        nodes = np.unique(self.faces_v_array.delete_rows(self.face_nodes, np.flatnonzero(self._non_boundary_mask())))
        X, Y = x[nodes], y[nodes]
        bx1, bx2 = bounding_box[0], bounding_box[0] + bounding_box[2]
        by1, by2 = bounding_box[1], bounding_box[1] + bounding_box[3]
        L = len(nodes)
        e = np.ones(L)
        dist = np.stack([np.abs(X - bx1), np.abs(X - bx2), np.abs(Y - by1), np.abs(Y - by2)], axis=0)
        Xo = np.stack([bx1 * e, bx2 * e, X, X], axis=0)
        Yo = np.stack([Y, Y, by1 * e, by2 * e], axis=0)
        idx = np.argmin(dist, axis=0)
        Xo, Yo = Xo[(idx, np.arange(L))], Yo[(idx, np.arange(L))]
        ido = np.arange(len(Xo)) + len(x)
        X = np.concatenate((x, Xo, [bx1, bx2, bx2, bx1]))
        Y = np.concatenate((y, Yo, [by1, by1, by2, by2]))
        P = len(x) + len(ido)

        sxl = ido[idx == 2][np.argsort(Xo[idx == 2])]
        sxh = ido[idx == 3][np.argsort(Xo[idx == 3])]
        syl = ido[idx == 0][np.argsort(Yo[idx == 0])]
        syh = ido[idx == 1][np.argsort(Yo[idx == 1])]

        N1 = np.concatenate((n1, nodes, np.append(sxl, [P+ 1]), np.append(sxh, [P+2]), np.append(syl, [P+3]), np.append(syh, [P+2])))
        N2 = np.concatenate((n2, ido,   np.append([P], sxl),    np.append([P+3], sxh), np.append([P], syl),   np.append([P+1], syh)))
        return EmbeddedGraph(X, Y, N1, N2)

    def extended_dual_graph(self, bounding_box=None, box_extend_factor=1.2):
        dual_graph = self.extended_graph(bounding_box=bounding_box,
                                         box_extend_factor=box_extend_factor).face_dual_graph()
        mapping = dual_graph.locate_faces(*self.coo())
        return dual_graph, mapping
