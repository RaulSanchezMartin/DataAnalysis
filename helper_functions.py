import csv
import numpy as np
from scipy.interpolate import interp1d

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
    rownum=0
    for row in file_1:
        if rownum == 0:
            header_1 = row
        else:
            matrix_1.append(row)
        rownum += 1
    return matrix_1, header_1
    
def read_txt(input_file):
    """
    Function that reads a .txt file.
    One argument is required by this function:
        *Name of the input file (or full root): input_file
    The output is a matrix
    """
    file_full_root_1 = input_file
    file_1=open(file_full_root_1, 'rb')
    matrix_1=[]
    for row in file_1:
        matrix_1.append(row.split())
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
    
def reduce_curve(input_name,delimiter,initial_point,
    final_point, number_points):
    """
    Function that takes a initial curve of x,y points
    and returns a equally-spaced curve interpolated 
    from the initial one.
    Parameters:
        *input_name: Name of the input file where the 
            initial curve is stored
        *delimiter: delimiter of the input file
        *initial point: initial x coordinate of the 
            returned curve
        *final point: final x coordinate of the 
            returned curve 
        *number_points: number of points of the 
            returned curve
    """
    initial_curve=read_data(input_name,delimiter)
    initial_curve=str_to_float_mat(initial_curve)
    initial_curve2 = np.asarray(initial_curve)
    x=initial_curve2[:,0]
    y=initial_curve2[:,1]
    fit_curve=interp1d(x, y)
    x_fit = np.linspace(initial_point,final_point,number_points)
    fitted_curve=[]
    for _x in x_fit:
        fitted_curve.append([])
        fitted_curve[-1].append(_x)
        fitted_curve[-1].append(float(fit_curve(_x)))   
    return fitted_curve

def dif_curves(curve_1,curve_2):
    """
    Function that computes the difference between two curves
    These two curves must have the same x coordinates
    Inputs:
        *curve_1: Name of the input file where the 
            curve_1 is stored
        *curve_2: Name of the input file where the 
            curve_2 is stored
    Two outputs are returned:
        *output_1: difference curve_1
        *output_2: average difference
    """
    dif_curve=[]
    for _i in range(len(curve_1)):
        dif_curve.append([])
        dif = abs(float(curve_1[_i][1])-float(curve_2[_i][1]))
        dif_curve[-1].append(float(curve_1[_i][0]))
        dif_curve[-1].append(dif)
    dif_array=np.asarray(dif_curve)
    average_dif = [float(sum(dif_array[:,1]))/len(dif_curve)]
    return dif_curve, average_dif
    
def plot_figure(input_name,extension,xlabel,ylabel):
    """
    Function that save the figure of x-y plot.
    Input data:
        *Name of the file where the row data is stored
            :input_name (without extension). The data 
            should be specified as a matrix
        *Extension of the input file: extension
        *Label of the x axis: xlabel
        *Label of the y axis: ylabel
    You must import the matplotlib.pyplot modulue as plt
    """
    data=str_to_float_mat(read_txt(input_name+extension))
    x=[row[1] for row in data]
    y=[row[0] for row in data]
    plt.plot(x,y,'-b')
    plt.title(input_name, fontsize=20)
    plt.ylabel(xlabel, fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.savefig(input_name+'.png')
    plt.close()
    
    
def detect_files(full_root,extension):
    """
    Function that detect and returns the a list of the name of 
    the files (without extension) located at some specific root.
    Inputs:
        *Root of the directory where the files are stored: full_root
        *Extension of the files to detect: extension
    You must import the os module
    """
    files=[]
    for file in os.listdir(full_root):
        if file.endswith(extension):
            for _i in range(len(extension)):
                file=file[:-1]
            files.append(file)
    return files

def selective_extract(data,index_select):
    """
    Function that takes as an input a matrix (data).
    It returns as a dictionary which includes different
    submatrixs of the inital matrix. This submatrixs
    are obtained based on the value of one column (index_select)
    """
    select_set=set([])
    for row in data:
        select_set.add(int(row[index_select]))
    data_select = {}
    for i in list(select_set):
        data_select[i]=[]
    for row in data:
        index=int(row[index_select])
        data_select[index].append(row)
    return data_select
    
def selective_extract_str(data,index_select):
    """
    Function that takes as an input a matrix (data).
    It returns as a dictionary which includes different
    submatrixs of the inital matrix. This submatrixs
    are obtained based on the name of one column (index_select)
    """
    select_set=set([])
    for row in data:
        select_set.add(row[index_select])
    data_select = {}
    for i in list(select_set):
        data_select[i]=[]
    for row in data:
        data_select[row[index_select]].append(row)
    return data_select

def add_order(data,index_order):
    """
    Function that takes as an input a matrix (data).
    It returns the same matrix, but with an additional 
    column that indicates the order of each row depending
    on the value of one row (index_order)
    """
    list_to_order=[]
    for row in data:
        list_to_order.append(float(row[index_order]))
    indexs_ordered=sorted(range(len(list_to_order)), key=lambda k: list_to_order[k])
    for idx in range(len(data)):
        data[idx].append(indexs_ordered[idx]+1)
    return data

    
def extract_submatrix(data,ini_i,ini_j,final_i,final_j):
    """
    Function that takes a matrix as an input(data)
    and returns a submatrix extrated from this initial
    matrix only including specific rows and columns, that
    are indicated by four indexs:
        *ini_i: first row
        *final_i: final row
        *ini_j: first column
        *final_j: last column
    """
    final_matrix=[]
    for idx in range(ini_i,final_i+1):
        final_matrix.append(data[idx][ini_j:final_j+1])
    return final_matrix
    
def similarity_matrix(data,row_col):
    """
    Function that takes as an a matrix (data). It returns
    what is called a "similarity matrix" paying attention 
    to the columns (row_col=1) or to the rows (row_col=0)
    """
    if row_col==1:
        simi_matrix=[[0 for jdx in range(len(data[0]))] for idx in range(len(data[0]))]
        columns=[]
        for idx in range(len(data[idx])):
            print "First", idx
            column = (np.asarray(data))[:,idx]
            columns.append(list(column))
        for idx in range(len(data)):
            print "second", idx
            for jdx in range(idx,len(data[idx])):
                if not idx==jdx:
                    counter1=0
                    counter2=0
                    for kdx in range(len(columns)):
                        counter1 += 1
                        if columns[idx][kdx]==columns[jdx][kdx]:
                            counter2 += 1
                    simi_matrix[idx][jdx]=float(counter2)/float(counter1)
                    simi_matrix[jdx][idx]=simi_matrix[idx][jdx]
                else:
                    simi_matrix[idx][jdx]=1.0
    return simi_matrix
    
def remove_spaces_list(header):
    """
    Function that removes of the leading and ending spaces
    of each element of a list
    """
    output = []
    for item in header:
        output.append(item.strip())
    return output
    
    
def norm_vector(vector):
    """
    Function that returns the unit vector corresponding to a inital
    vector
    """
    norm = math.sqrt(sum([element**2 for element in vector]))
    return [element/norm for element in vector]

def shear_stress_global(flag,m,s):
    """
    Function that calculates the on a slip/twin plane in a slip/twin direction
    applying a pure tension or compression unit stress in z
    Inputs:
            flag ---> 1 for tension, 0 otherwise
            m ---> unit vector normal to the slip/twin plane (global coordinates)
            s ---> unit vector in the slip/twin direction (global coordiantes)
    """ 
        
    m=np.array(norm_vector(m))[np.newaxis]
    s=np.array(norm_vector(s))[np.newaxis]
    
    if flag==1:
        return np.kron(s,m.T)[-1,-1]
    else:
        return -np.kron(s,m.T)[-1,-1]

def rotation_matrix(m_tw):
    """
    Function that, given the normal vector to a twin plane "m_tw",
    returns the rotation matrix corresponding to that vector
    which has undergone a mirror-like rotation
    """
    Q11=2*m_tw[0]**2-1
    Q22=2*m_tw[1]**2-1
    Q33=2*m_tw[2]**2-1
    Q12=2*m_tw[0]*m_tw[1]
    Q13=2*m_tw[0]*m_tw[2]
    Q23=2*m_tw[1]*m_tw[2]
    return [[Q11,Q12,Q13],[Q12,Q22,Q23],[Q13,Q23,Q33]]
    
def rotation(vec, rot_mat):
    """
    Function that, given a vector 1x3 "vec" and a 3x3 rotation matrix 
    "mat_rot", returns the vector rotated
    """
    rot_vec1 = rot_mat[0][0]*vec[0]+rot_mat[0][1]*vec[1]+rot_mat[0][2]*vec[2]
    rot_vec2 = rot_mat[1][0]*vec[0]+rot_mat[1][1]*vec[1]+rot_mat[1][2]*vec[2]
    rot_vec3 = rot_mat[2][0]*vec[0]+rot_mat[2][1]*vec[1]+rot_mat[2][2]*vec[2]
    return [rot_vec1, rot_vec2, rot_vec3]
    
def angle_two_units_vectors(vec1, vec2):
    """
    Function that, given two units vectors "vec1" and "vec2",
    returns, in degrees, the angle between them
    """
    return math.acos(vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2])*180/math.pi
    
    
def direction_change_hkil_hkl (UVTW):
    """
    Function that transforms the U,V,T,W coordinates 
    of a direction given in the hkil coordinate systems
    into the u,v,w coordinates of the hkl coordinate system
    """
    u = UVTW[0] - UVTW[2]
    v = UVTW[1] - UVTW[2]
    w = UVTW[3]
    return [u, v, w]

def plane_change_hkil_hkl (UVTW):
    """
    Function that transforms the U,V,T,W coordinates 
    of a plane given in the hkil coordinate systems
    into the u,v,w coordinates of the hkl coordinate system
    """
    u = UVTW[0]
    v = UVTW[1]
    w = UVTW[3]
    return [u, v, w]
    
def uvw_to_xyz_change(uvw,OM):
    """
    Function that transforms the u,v,w coordinates 
    of a vector given in the hkl coordinate systems
    into the x,y,z coordinates of the cartesian coordinate system
    """
    uvw = np.array(uvw)
    x_raw = np.inner(np.asarray(OM[0]), uvw)
    y_raw = np.inner(np.asarray(OM[1]), uvw)
    z_raw = np.inner(np.asarray(OM[2]), uvw)
    module = math.sqrt(x_raw**2+y_raw**2+z_raw**2)
    return [x_raw/module, y_raw/module, z_raw/module]
    
    
def plane_uvtw_hkil(hkil, c_a):
    """
    Function that, given the hkil Miller-Bravais indexs (hkil)
    and the c/a ratio (c_a) returns the normal vector
    """
    parameter = math.sqrt(float(2)/3)*c_a
    U = hkil[0]
    V = hkil[1]
    T = hkil[2]
    W = hkil[3]/(parameter**2)
    return [U,V,T,W]
    
def trans_euler_angles(phi1, PHI, phi2):
    """
    Function that, given the three eluer angles (phi1, PHI, phi2)
    returns the projection of the corresponding x and y axis in the 
    grlobal system
    """
    sin=math.sin
    cos=math.cos
    pi=math.pi

    phi1=phi1*2*pi/360
    PHI=PHI*2*pi/360
    phi2=phi2*2*pi/360
    
    a = [[0 for i in xrange(3)] for i in xrange(2)]
    
    a[0][0]=cos(phi2)*cos(phi1)-sin(phi1)*sin(phi2)*cos(PHI)
    a[0][1]=cos(phi2)*sin(phi1)+cos(phi1)*sin(phi2)*cos(PHI)
    a[0][2]=sin(PHI)*sin(phi2)
    
    a[1][0]=-sin(phi2)*cos(phi1)-sin(phi1)*cos(phi2)*cos(PHI)
    a[1][1]=-sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2)*cos(PHI)
    a[1][2]=cos(phi2)*sin(PHI)
    
    return [a[0][0],a[0][1],a[0][2],a[1][0],a[1][1],a[1][2],0]
    
    
def replace_value(value,limits):
    """
    Function that replaces an input "value" to 
    other final value taking into account some limits.
    These "limits" are given as a list according to the 
    following syntax:
        limits = [limit_1, limit_2, ..., limit_i]
        limit_i[[lowest_value, highest_value],'value_to_replace']
    The function maps limit by limit, doing the following:
        if lowest_value <= value <= highest_value:
            final_value --> 'value_to_replace'
            break
    """
    new_value = value
    for limit in limits:
        lowest_value = float(limit[0][0])
        highest_value = float(limit[0][1])
        replace_value = limit[1]
        if lowest_value<=value<=highest_value:
            new_value = replace_value
            break
    return new_value
    
def map_matrix_replace_value(input_data,input_header,ini_jdx,fin_jdx):
    """
    Function that takes a input header and a input matrix (input_data). 
    In addition, it is required two more values: ini_jdx and fin_jdx.
    The function maps all the values of the input matrix. If the corresponding
    j indexs of that vale are ini_jdx <= j_index <= fin_jdx, the replace_value()
    function is applied
    """
    new_matrix = []
    new_matrix.append(input_header)
    for idx in range(len(input_data)):
        new_matrix.append([])
        for jdx in range(len(input_data[idx])):
            if ini_jdx <= jdx <= fin_jdx:
                new_matrix[-1].append(replace_value(float(input_data[idx][jdx]),limits))
            else:
                new_matrix[-1].append(input_data[idx][jdx])
    return new_matrix
    
    
def euler_angles_to_rotation_matrix(E1, E2, E3):
    """
    Function that, given three Euler angles E1 (phi1), E2 (PHI) and E3 (phi2),
    returns the corresponding rotation matrix
    """

    sin=math.sin
    cos=math.cos
    pi=math.pi
    
    E1=E1*2*pi/360
    E2=E2*2*pi/360
    E3=E3*2*pi/360
    
    a = [[0 for i in xrange(3)] for i in xrange(3)]
    
    s1 = sin(E1)
    c1 = cos(E1)
    s2 = sin(E2)
    c2 = cos(E2)
    s3 = sin(E3)
    c3 = cos(E3)
    
    a[0][0] = c1*c3 - c2*s1*s3
    a[0][1] = -c1*s3 - c2*c3*s1
    a[0][2] = s1*s2
    a[1][0] = c3*s1 + c1*c2*s3
    a[1][1] = c1*c2*c3 - s1*s3
    a[1][2] = -c1*s2
    a[2][0] = s2*s3
    a[2][1] = c3*s2
    a[2][2] = c2
    
    return a