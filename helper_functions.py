import csv
import numpy as np

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
        
def find_element_matrix(matrix,matrix_index,elements_list):
    """
    Function that iterates over the matrix[matrix_index]
    items searching for each element in the elements_list
    The input matrix is returned, but only including that rows
    that in which the matrix[matrix_index] item has been found 
    in elements_list
    """
    matrix_output = []
    for row in matrix:
        if row[matrix_index] in elements_list:
            matrix_output.append(row)
    matrix_output = np.array(matrix_output)
    return matrix_output
    
def distribution_list(list_input):
    """
    Function that returns the distribution of the elements of list
    A two dimensional array is returned according to the following scheme:
        *element[i]/% of times that element[i] appears in list
    """
    matrix_output = []
    set_list = set(list_input)
    set_list = list(set_list)
    set_list.sort()
    len_list = len(list_input)
    row_idx = 0
    for idx in range(len(set_list)):
        matrix_output.append([])
        count_item = list_input.count(set_list[idx])
        count_item_per = float(count_item)/len_list*100
        matrix_output[idx].append(set_list[idx])
        matrix_output[idx].append(count_item_per)
    return matrix_output
    