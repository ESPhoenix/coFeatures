## basic libraries
import numpy as np
import pandas as pd
## basic utilities
########################################################################################
# loads a pdb file as a dataframe
def pdb2df(pdbFile):
    columns = ['ATOM', 'ATOM_ID', 'ATOM_NAME', 'RES_NAME', 'CHAIN_ID',
                'RES_SEQ', 'X', 'Y', 'Z', 'OCCUPANCY', 'TEMP_FACTOR', 'ELEMENT']

    data = []
    with open(pdbFile, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_type = line[0:6].strip()
                atom_id = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:23].strip()
                if chain_id == '':
                    chain_id = None
                res_seq = int(line[23:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occupancy = float(line[54:60].strip())
                temp_factor = float(line[60:66].strip())
                element = line[76:78].strip()

                data.append([atom_type, atom_id, atom_name, res_name, chain_id,
                             res_seq, x, y, z, occupancy, temp_factor, element])

    return pd.DataFrame(data, columns=columns)
########################################################################################
# calculate euclidean distance between a row and a single point 
def calculateEuclideanDistance(row, point):
    xDiff = row['X'] - point[0]
    yDiff = row['Y'] - point[1]
    zDiff = row['Z'] - point[2]
    euclidean = np.sqrt(xDiff**2 + yDiff**2 + zDiff**2)
    
    return float(euclidean)
########################################################################################