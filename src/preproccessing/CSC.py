class CSCMatrix:

    def __init__(self, data, ind, ptr, shape, dtype = "d"):
        """
        Initializes a compressed sparse column (CSC) matrix
        param data: An iterable of the nonzero data elements
        param ind: An iterable of the nonzero data indices
        param ptr: An iterable of the nonzero data index pointers
        param shape: A tuple giving the (nrow, ncol)
        param dtype: The data type (defaults to double)
        """
        self.nrow, self.ncol = shape
        self.data = data
        self.ind = ind
        self.ptr = ptr

    def col_means(self):
        """
        Calculates and returns the means of the matrix columns.
        param self: An instance of CSCMatrix.
        returns: A list of numbers giving the means of each column.
        """
        mean_list = [0] * self.ncol
        for i in range(self.ncol):
            start = self.ptr[i]
            end = self.ptr[i + 1]
            mean_list[i] = sum(self.data[start:end]) / (end - start)
        return mean_list
    
print(CSCMatrix([33, 11, 55, 22, 44], [1,0,2,0,1], [0,1,3,3,5], (3,4)).col_means())
