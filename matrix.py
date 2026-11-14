from numbers import Real


class Matrix:
    def __init__(self, rows: int, cols: int, data=None):
        if rows < 0 or cols < 0:
            raise ValueError("Matrix dimensions must be non-negative.")

        self.rows = rows
        self.cols = cols
        if data is None:
            self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        else:
            if len(data) != rows or any(len(row) != cols for row in data):
                raise ValueError("Data dimensions do not match the matrix dimensions.")
            self.data = [list(row) for row in data]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, j = idx
            return self.data[i][j]

        return Vector(self.cols, self.data[idx])

    def __setitem__(self, idx, value):
        if isinstance(idx, tuple):
            i, j = idx
            self.data[i][j] = value
            return

        if isinstance(value, Matrix):
            if value.rows == 1 and value.cols == self.cols:
                self.data[idx] = value.data[0][:]
            elif value.rows == self.cols and value.cols == 1:
                self.data[idx] = [value.data[j][0] for j in range(value.rows)]
            else:
                raise ValueError("Invalid row dimensions.")
        elif isinstance(value, Real) and self.cols == 1:
            self.data[idx] = [value]
        elif isinstance(value, (list, tuple)) and len(value) == self.cols:
            self.data[idx] = list(value)
        else:
            raise ValueError("Invalid row dimensions.")

    def __str__(self):
        return "\n".join("\t".join(map(str, row)) for row in self.data)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Incompatible matrix dimensions for multiplication.")
            result = Matrix(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    result[i, j] = sum(
                        self.data[i][k] * other.data[k][j]
                        for k in range(self.cols)
                    )
            return result

        if isinstance(other, Real):
            return Matrix(
                self.rows,
                self.cols,
                [[value * other for value in row] for row in self.data],
            )

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Real):
            return self * other
        return NotImplemented

    def __neg__(self):
        return self * -1

    def __add__(self, other: "Matrix"):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition.")
        return Matrix(
            self.rows,
            self.cols,
            [
                [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ],
        )

    def __sub__(self, other: "Matrix"):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for subtraction.")
        return Matrix(
            self.rows,
            self.cols,
            [
                [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ],
        )

    def transpose(self):
        return Matrix(
            self.cols,
            self.rows,
            [
                [self.data[i][j] for i in range(self.rows)]
                for j in range(self.cols)
            ],
        )

    def __iter__(self):
        return MatrixIterator(self)


class Vector(Matrix):
    """A column vector with shape size × 1."""

    def __init__(self, size: int, data=None):
        if data is None:
            column = [[0.0] for _ in range(size)]
        else:
            values = list(data)
            if len(values) == 1 and isinstance(values[0], (list, tuple)):
                values = list(values[0])
            if len(values) != size or any(isinstance(value, (list, tuple)) for value in values):
                raise ValueError("Vector data length does not match its size.")
            column = [[value] for value in values]
        super().__init__(size, 1, column)


class MatrixIterator:
    def __init__(self, matrix: Matrix):
        self.matrix = matrix
        self.current_row = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_row >= self.matrix.rows:
            raise StopIteration

        row = self.matrix.data[self.current_row]
        self.current_row += 1
        if self.matrix.cols == 1:
            return row[0]
        return Vector(self.matrix.cols, row)
