import csv

def read_data(input_name, delimiter):
    """
    Function that reads a .csv file.
    Two arguments are required by this function:
        *Name of the input file (or full root): input_name
        *Delimiter of the .csv file: delimiter
    The output is a matrix
    """
    file_1 = csv.reader(open(input_name), delimiter = delimiter)
    matrix_1=[] 
    for row in file_1:
        if row != []:
            matrix_1.append(row)
    return matrix_1
    
def str_to_float_mat(matrix):
    """
    Function that converts a matrix of 
    strings into a matrix of floats
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = float(matrix[i][j])
    return matrix
    
def str_to_int_mat(matrix):
    """
    Function that converts a matrix of 
    strings into a matrix of floats
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = int(matrix[i][j])
    return matrix
    
def write_data(output_name, output_data, delimiter):
    """
    Function that writes a .csv file.
    Three arguments are required by this function:
        *Data to be export (in a matrix-shape): output_data
        *Name of the output file (or full root): output_name
        *Delimiter of the .csv file: delimiter
    """
    with open(output_name, "wb") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(output_data)
