import numpy as np
import scipy.sparse


class VarRowArray:
    """
    Array structure representing 2D array with variable row lengths.
    The rows are called the "inner" dimension, the columns the "outer".
    One supplies the length of each row with counts.

    example:
    counts        [2 3 1 2]   represents [[., .], [., ., .], [.], [., .]] array
    columns       [0 1 0 1 2 0 0 1]
    rows          [0 0 1 1 1 2 3 3]
    row_length    [2 2 3 3 3 1 2 2]
    __len__()     8
    get_item([1, 0, 2, 1, 3], [1, 1, 0, 2, 1]) -> [3, 1, 5, 4, 7]
    roll(axis=1)  [1 0 4 2 3 5 7 6]
    roll(axis=0)  [6 7 0 1 2 3 4 5]

    """

    def __init__(self, counts):
        """

        :param counts:
        """
        self.counts = np.array(counts, dtype=int)
        self.cum_counts = np.cumsum(self.counts) - self.counts
        self.total = np.sum(self.counts)

    def __len__(self):
        return self.total

    def row_count(self):
        """
        compute row count

        :return:
        """
        return len(self.counts)

    def columns(self, rows=None):
        # if rows is None; returns column indices for all rows
        if rows is None:
            return np.arange(len(self)) - np.repeat(self.cum_counts, self.counts)
        else:
            c = self.counts[rows]
            return np.arange(np.sum(c)) - np.repeat(np.cumsum(c) - c, c)

    def row_ranges(self):
        return self.cum_counts, self.cum_counts + self.counts

    def sum(self, data, axis=1):
        if axis == 0:
            raise ValueError("sum along axis=0 not implemented")
        if axis == 1:
            s_data = np.append([0], np.cumsum(data))
            return s_data[self.cum_counts + self.counts] - s_data[self.cum_counts]
        raise ValueError("axis must be 0 or 1")

    def rows(self):
        return np.repeat(np.arange(len(self.counts)), self.counts)

    def at_out_index(self, data):
        return np.repeat(data, self.counts)

    def row_length(self):
        return np.repeat(self.counts, self.counts)

    def get_item(self, rows=None, columns=None):
        if rows is not None:
            rows = np.arange(self.row_count())[rows]
        if rows is not None and columns is not None:
            return np.array(columns, dtype=int) + self.cum_counts[rows]
        if columns is None:
            columns = self.columns(rows=rows)
        if rows is None:
            rows = self.rows()
        else:
            rows = np.repeat(np.array(rows), self.counts[rows])

        return columns + self.cum_counts[rows]

    def delete_rows(self, data, rows):
        return np.delete(data, self.get_item(rows=rows))

    def roll(self, n=1, axis=1):
        if axis==0:
            if n >= 0:
                return np.roll(np.arange(self.total), np.sum(self.counts[-n:]))
            else:
                return np.roll(np.arange(self.total), -np.sum(self.counts[:-n]))
        if axis==1:
            return self.get_item(columns=(self.columns() - n) % self.row_length())
        raise ValueError("axis must be 0 or 1")

    def to_csr(self, cols, data, num_cols=None):
        if num_cols is None:
            num_cols = np.max(cols)
        indptr = np.append(self.cum_counts, [self.total])
        return scipy.sparse.csr_matrix((data, cols, indptr), shape=(self.row_count(), num_cols))

    def to_list(self, data):
        return [[data[self.cum_counts[row] + col] for col in range(self.counts[row])] for row in range(self.row_count())]

    def permute_rows(self, permutation, data):
        permutation, data = np.array(permutation, dtype=int), np.array(data)
        return data[self.get_item(rows=permutation)]

    def merge(self, other):
        # returns sorter;
        # out = np.zeros(len(self) + len(other))
        # out[sorter] = np.append(data_self, data_other)
        t = self.counts + other.counts
        cum_t = np.cumsum(t) - t
        i1 = self.columns() + np.repeat(cum_t, self.counts)
        i2 = other.columns() + np.repeat(cum_t + self.counts, other.counts)
        return np.append(i1, i2)



if __name__ == "__main__":

    x = VarRowArray([2, 3, 1, 2])
    print(len(x))
    print(x.row_count())
    print(x.columns())
    print(x.rows())
    print(x.roll())
    print(x.roll(axis=0))
    print(x.get_item([1, 0, 2, 1, 3], [1, 1, 0, 2, 1]))
    print(x.to_list([12, 34, 13, 24, 56, 54, 32, 21]))

    data = [2, 6, 4, 5, 4, 6, 7, 8]
    perm = [2, 1, 3, 0]
    print(x.permute_rows(perm, data))
