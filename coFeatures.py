########################################################################################
from genericpath import isfile
from multiprocessing import process
import os
from os import path as p
import numpy as np
import pandas as pd
import multiprocessing as mp
import importlib
import argpass
import multiprocessing_logging
import logging

########################################################################################
## basic utilities
def pdb2df(pdbFile):
    columns = ['ATOM', 'ATOM_ID', 'ATOM_NAME', 'RES_NAME', 'CHAIN_ID', 'RES_SEQ', 'X', 'Y', 'Z', 'OCCUPANCY', 'TEMP_FACTOR', 'ELEMENT']

    data = []
    with open(pdbFile, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_type = line[0:6].strip()
                atom_id = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                if chain_id == '':
                    chain_id = None
                res_seq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occupancy = float(line[54:60].strip())
                temp_factor = float(line[60:66].strip())
                element = line[76:78].strip()

                data.append([atom_type, atom_id, atom_name, res_name, chain_id, res_seq, x, y, z, occupancy, temp_factor, element])

    return pd.DataFrame(data, columns=columns)

def calculateEuclideanDistance(row, point):
    xDiff = row['X'] - point[0]
    yDiff = row['Y'] - point[1]
    zDiff = row['Z'] - point[2]
    euclidean = np.sqrt(xDiff**2 + yDiff**2 + zDiff**2)
    
    return float(euclidean)

def ifnotmkdir(dir):
    if not p.isdir(dir):
        os.mkdir(dir)
    return dir

########################################################################################
# get inputs
def read_inputs():
    global inputDir, outputDir, aminoAcidTable, cofactorPresent,cofactorNames
    global keyAtomsDict, orbAtomsDict, cloudAtomsDict, orbRange, cloudRange
    parser = argpass.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    configName=args.config
    if  args.config == None:
        print('No config file name provided.')
        exit()
    configName = p.splitext(configName)[0]

    inputScript = importlib.import_module(configName)
    globals().update(vars(inputScript))
########################################################################################

def initialiseAminoAcidInformation(aminoAcidTable):
    AminoAcidNames = ["ALA","ARG","ASN","ASP","CYS",
                      "GLN","GLU","GLY","HIS","ILE",
                      "LEU","LYS","MET","PHE","PRO",
                      "SER","THR","TRP","TYR","VAL"]  

    # read file with amino acids features
    aminoAcidProperties = pd.read_csv(
        aminoAcidTable, sep="\t", index_col=1
    )
    aminoAcidProperties.index = [el.upper() for el in aminoAcidProperties.index]
    aminoAcidProperties = aminoAcidProperties.iloc[:, 1:]

    return AminoAcidNames, aminoAcidProperties

########################################################################################
def getPdbList(dir):
    pdbList=[]
    idList=[]
    for file in os.listdir(dir):
        fileData = p.splitext(file)
        if fileData[1] == '.pdb':
            idList.append(fileData[0])
            pdbList.append(p.join(dir,file))
    return idList, pdbList

########################################################################################
def find_cofactor(cofactorNames,pdbDf,protName):
    # quick check to see what cofactor is present in a pdb file
    # Only works if there is one cofactor - will break with two
    # returns a dictionary of true/false for all cofactors present in confactorNames input variable
    cofactorCheck = {}
    for cofactor in cofactorNames:
        if pdbDf["RES_NAME"].str.contains(cofactor).any():
            cofactorCheck.update({cofactor:True})
        else:
            cofactorCheck.update({cofactor:False})
    # count number of cofactors if interest present
    countTrue = sum(value for value in cofactorCheck.values() if value)
    # if cofactor count is not 1, throw an error message, then this structure will be skipped
    # if cofactor count is 1, identify the cofactor and retirn as a variable
    cofactorCountWrong = False
    if countTrue == 0:
        print(f"no cofactor found in {protName}")
        cofactorCountWrong = True
        return None, cofactorCountWrong
    elif countTrue > 1:
        print(f'{countTrue} cofactors found in {protName}, need only one!')
        cofactorCountWrong = True
        return None, cofactorCountWrong
    elif countTrue == 1:
        for key, value in cofactorCheck.items():
            if value:
                cofactorName = key
                return cofactorName, cofactorCountWrong

########################################################################################
def get_key_atom_coords(pdbDf,keyAtomsDict,cofactorName):
    keyAtoms = keyAtomsDict.get(cofactorName,[])

    keyAtomCoords=pd.DataFrame(columns=["X","Y","Z"], index=keyAtoms)

    cofactorRows = pdbDf[pdbDf["RES_NAME"] == cofactorName]
    for atomId in keyAtoms:
        xCoord = cofactorRows.loc[cofactorRows["ATOM_NAME"] == atomId, "X"].values
        yCoord = cofactorRows.loc[cofactorRows["ATOM_NAME"] == atomId, "Y"].values
        zCoord = cofactorRows.loc[cofactorRows["ATOM_NAME"] == atomId, "Z"].values
        keyAtomCoords.loc[atomId] = [xCoord,yCoord,zCoord]
    return keyAtomCoords
########################################################################################

def get_orb_center(orbAtomsDict,cofactorName, pdbDf):
    # get orbAboms that correspond to cofactor present in this pdbfile
    orbAtoms = orbAtomsDict.get(cofactorName,[])

    cofactorRows = pdbDf[pdbDf["RES_NAME"] == cofactorName]
    orbRows = cofactorRows[cofactorRows["ATOM_NAME"].isin(orbAtoms)]
    
    orbCenter=[]
    orbCenter.append(orbRows["X"].mean())
    orbCenter.append(orbRows["Y"].mean())
    orbCenter.append(orbRows["Z"].mean())

    return orbCenter
########################################################################################

def get_cloud_atom_coords(cloudAtomsDict,cofactorName, pdbDf):
    cloudAtoms = cloudAtomsDict.get(cofactorName,[])

    cofactorRows = pdbDf[pdbDf["RES_NAME"] == cofactorName]
    cloudRows = cofactorRows[cofactorRows["ATOM_NAME"].isin(cloudAtoms)]
    
    cloudCoordsDf=cloudRows[["ATOM_NAME","X","Y","Z"]]
    cloudCoordsDf.set_index("ATOM_NAME",inplace=True)

    return cloudCoordsDf
    
########################################################################################
def orb_count(orbCenter,pdbDf,aminoAcidNames,orbValue,proteinName,cofactorName):
    # initialise amino acid count dataframe (better as a dict?)
    columnNames=[]
    columnNames.append(f'orb.total')
    for aminoAcid in aminoAcidNames:
        columnNames.append(f'orb.{aminoAcid}')
    for element in ["C","N","O"]:
        columnNames.append(f'orb.{element}')
    columnNames=sorted(columnNames)
    orbCountDf = pd.DataFrame(columns=columnNames,index=[proteinName])
    # remove cofactor from pdbDf
    pdbDf = pdbDf[pdbDf["RES_NAME"] != cofactorName].copy()
    # get count of each AA within orbValue of orbycenter
    # get distacnes to orbycenter
    pdbDf.loc[:, "orb_DISTANCES"] = pdbDf.apply(calculateEuclideanDistance, point=orbCenter, axis=1)
    # filter main dataframe by atoms within orbValue range of orbycenter
    orbDf = pdbDf[pdbDf["orb_DISTANCES"] <= orbValue].copy()
    if orbDf.empty:
        orbCountDf.iloc[0] = 0
        return orbCountDf
    # count elements [Carbon, Nitrogen, Oxygen]
    for element in ["C","N","O"]:
        try:
            orbCountDf.loc[:,f'orb.{element}'] = orbDf["ELEMENT"].value_counts()[element]
        except:
            orbCountDf.loc[:, f'orb.{element}'] = 0
 
    #  get unique residues to avoid over-counting
    orbDf_uniqueRes=orbDf.drop_duplicates(subset=["RES_SEQ"])
    # add individial and total amino acid counts to count featues dataframe
    orbTotal=0
    for aminoAcid in aminoAcidNames:
        try: 
            orbCountDf.loc[:,f'orb.{aminoAcid}'] = orbDf_uniqueRes["RES_NAME"].value_counts()[aminoAcid]
        except:
            orbCountDf.loc[:,f'orb.{aminoAcid}'] = 0
        orbTotal+=orbCountDf.loc[:,f'orb.{aminoAcid}']
    orbCountDf.loc[:,"orb.total"] = orbTotal
    return orbCountDf
########################################################################################
def cloud_count(cloudCoordsDf,pdbDf,aminoAcidNames,cloudValue,proteinName):
    # initialise amino acid count dataframe (better as a dict?)
    columnNames=[]
    columnNames.append(f'cloud.total')
    for aminoAcid in aminoAcidNames:
        columnNames.append(f'cloud.{aminoAcid}')
    for element in ["C","N","O"]:
        columnNames.append(f'cloud.{element}')
    columnNames=sorted(columnNames)
    cloudCountDf = pd.DataFrame(columns=columnNames,index=[proteinName])
    # get count of each AA within cloudValue of any atom in cloud
    # for each atom in cloud...
    for i, (x,y,z) in cloudCoordsDf.iterrows():
        # calculate distance to atom and write to its own column
        columnName=f'cloud_dist_atom_{i}'
        pdbDf.loc[:,columnName]=pdbDf.apply(calculateEuclideanDistance, point=[x,y,z], axis =1)
    # identify cloud columns in pdbDf
    cloudCols = [col for col in pdbDf.columns if col.startswith('cloud_dist_atom_')]
    # filter main dataframe by atoms within cloudValue range of cloud atoms
    withincloudRange = pdbDf[cloudCols].lt(cloudValue).any(axis=1)
    cloudDf=pdbDf[withincloudRange]
    # if no atoms are within range, return empty features dataframe
    if cloudDf.empty:
        cloudCountDf.iloc[0] = o
        return cloudCountDf
    # count elements [Carbon, Nitrogen, Oxygen]
    for element in ["C","N","O"]:
        try:
            cloudCountDf.loc[:,f'cloud.{element}'] = cloudDf["ELEMENT"].value_counts()[element]
        except KeyError as e:
            cloudCountDf.loc[:, f'cloud.{element}'] = 0
    # get unique residues only
    cloudDf_uniqueRes= cloudDf.drop_duplicates(subset=["RES_SEQ"])
    cloudTotal=0
    # add individial and total amino acid counts to count featues dataframe
    for aminoAcid in aminoAcidNames:
        if aminoAcid in cloudDf["RES_NAME"].unique():
            cloudCountDf.loc[:,f'cloud.{aminoAcid}'] = cloudDf_uniqueRes["RES_NAME"].value_counts()[aminoAcid]
        else:
            cloudCountDf.loc[:,f'cloud.{aminoAcid}'] = 0
        cloudTotal+=cloudCountDf[f'cloud.{aminoAcid}']
    cloudCountDf["cloud.total"] = cloudTotal
    return cloudCountDf
        
########################################################################################
def protein_count(pdbDf,aminoAcidNames,proteinName):
    # initialise amino acid count dataframe (better as a dict?)
    columnNames=[]
    columnNames.append(f'protein.total')
    for aminoAcid in aminoAcidNames:
        columnNames.append(f'protein.{aminoAcid}')
    for element in ["C","N","O"]:
        columnNames.append(f'protein.{element}')
    columnNames=sorted(columnNames)
    proteinCountDf = pd.DataFrame(columns=columnNames,index=[proteinName])

    # count elements [Carbon, Nitrogen,Oxygen]
    for element in ["C","N","O"]:
        try:
            proteinCountDf.loc[:,f'protein.{element}'] = pdbDf["ELEMENT"].value_counts()[element]
        except KeyError:
            proteinCountDf.loc[:, f'protein.{element}'] = 0
    # get count of each AA in the whole protein
    # get unique residue IDs only
    pdbDf_uniqueRes=pdbDf.drop_duplicates(subset=["RES_SEQ"])
    proteinTotal=0
    # add individial and total amino acid counts to count featues dataframe
    for aminoAcid in aminoAcidNames:
        if aminoAcid in pdbDf_uniqueRes["RES_NAME"].unique():
            proteinCountDf.loc[:,f'protein.{aminoAcid}'] = pdbDf_uniqueRes["RES_NAME"].value_counts()[aminoAcid]
        else:
            proteinCountDf.loc[:,f'protein.{aminoAcid}'] = 0
        proteinTotal+=proteinCountDf[f'protein.{aminoAcid}']
    proteinCountDf["protein.total"] = proteinTotal
    return proteinCountDf 

########################################################################################
def cloud_orb_protein_properties(orbCountDf,cloudCountDf,proteinCountDf,aminoAcidProperties,aminoAcidNames,proteinName):
    # initialise pandas dataframe
    columnNames=[]
    for property in aminoAcidProperties.columns:
        columnNames.append(f'orb.{property}')
        columnNames.append(f'cloud.{property}')
        columnNames.append(f'protein.{property}')
    propertiesDf=pd.DataFrame(columns=columnNames,index=[proteinName])
    
    for region, countDf in zip(["orb","cloud","protein"],[orbCountDf,cloudCountDf,proteinCountDf]):
        for property in aminoAcidProperties:
            propertyValue=0
            for aminoAcid in aminoAcidNames:
                try:
                    aaCount = countDf.at[proteinName,f"{region}.{aminoAcid}"]
                except KeyError:
                    aaCount = 0
                aaPropertyvalue = aminoAcidProperties.at[aminoAcid,property]
                value = aaCount * aaPropertyvalue
                propertyValue += value 
            try:
                totalAminoAcids=countDf.at[proteinName,f'{region}.total']
            except KeyError:
                totalAminoAcids=0
            if not totalAminoAcids == 0:
                propertyValue = propertyValue / totalAminoAcids
            propertiesDf[f'{region}.{property}'] = propertyValue
    return propertiesDf
########################################################################################
def get_key_atom_features(keyAtomCoords, pdbDf, aminoAcidNames, aminoAcidProperties,proteinName,cofactorName):
    # initialse dataframe
    columnNames=[]
    for keyAtomId, (x,y,z) in keyAtomCoords.iterrows():
        for property in aminoAcidProperties:
            columnNames.append(f'1_{keyAtomId}.{property}')
            columnNames.append(f'3_{keyAtomId}.{property}')
    keyAtomFeaturesDf = pd.DataFrame(columns=columnNames, index=[proteinName])
    # remove cofactor from pdbDf
    pdbDf = pdbDf[pdbDf["RES_NAME"] != cofactorName].copy()
    # loop through key atom coords
    for keyAtomId, (x,y,z) in keyAtomCoords.iterrows():
        # calculate distance for each atom in pdb dataframe
        columnName=f'{keyAtomId}_distance'
        pdbDf.loc[:,columnName]=pdbDf.apply(calculateEuclideanDistance, point=[x,y,z], axis =1)
        # Get the unique residue sequences
        uniqueResidues = pdbDf['RES_SEQ'].unique()
        # Iterate over each unique residue and find the nearest point
        nearestPointIndecies=[]
        for residueSeq in uniqueResidues:
            residueData = pdbDf[pdbDf['RES_SEQ'] == residueSeq]
            nearestPointIdx = residueData[columnName].idxmin()
            nearestPointIndecies.append(nearestPointIdx)
        # make a subset dataframe   
        nearestPointsResidues = pdbDf.loc[nearestPointIndecies]
        # Reset the index of the resulting DataFrame
        nearestPointsResidues.reset_index(drop=True, inplace=True)
        # sort by distance to key atom
        nearestResiduesSorted = nearestPointsResidues.sort_values(by=columnName, ascending=True)
        # get nearest one and three 
        nearestOneResidue=''.join(nearestResiduesSorted.head(1)["RES_NAME"].to_list())
        nearestThreeResidues=nearestResiduesSorted.head(3)["RES_NAME"].to_list()
        # update features dataframe for single nearest amino acid
        for property in aminoAcidProperties:
            propertyValue = aminoAcidProperties.at[nearestOneResidue,property]
            keyAtomFeaturesDf.loc[proteinName,f'1_{keyAtomId}.{property}'] = propertyValue
        # update features dataframe for three nearest amino acids
        for property in aminoAcidProperties:
            propertyValue=0
            for aminoAcid in nearestThreeResidues:
                value = aminoAcidProperties.at[aminoAcid,property]
                propertyValue += value 
            propertyValue = propertyValue / 3
            keyAtomFeaturesDf.loc[proteinName,f'3_{keyAtomId}.{property}'] = propertyValue
    return keyAtomFeaturesDf


########################################################################################
def process_pdbs(jobDetails):
    # unpack jobData into individual variables
    orbValue,cloudValue,jobProgress,idList, pdbList,outDir,aminoAcidNames,aminoAcidProperties = jobDetails
    # initialise output file, check if it exists, if so, skip iteration.
    outputCsv=p.join(outDir, f'orb_{str(orbValue)}_cloud_{str(cloudValue)}_features.csv')
    if p.isfile(outputCsv):
        print(f"-->\t{jobProgress}\tSkipping 'orb_{str(orbValue)}_cloud_{str(cloudValue)}_features.csv")
        return
    # initialise a list for stocloud dataframes of features 
    print(f"-->\t{jobProgress}\tMaking coFeatures for orb_{str(orbValue)}_cloud_{str(cloudValue)}_features.csv...")
    featuresList=[]
    # for each permute of orb + cloud, loop through all pdb files in pdbDir

    for id, pdbFile in zip(idList, pdbList):
        pdbDf=pdb2df(pdbFile)
        cofactorName, cofactorCountWrong = find_cofactor(pdbDf=pdbDf,
                                                          cofactorNames=cofactorNames, 
                                                          protName=id)
        if cofactorCountWrong:
            continue
        keyAtomCoords = get_key_atom_coords(pdbDf,keyAtomsDict,cofactorName)
        # get coords of  orb center as a 3 element list ([x,y,z])
        orbCenter = get_orb_center(orbAtomsDict=orbAtomsDict,
                                   cofactorName=cofactorName,
                                   pdbDf=pdbDf)
        # get coords of heteroatoms in isoaloxizine cloud as a pandas dataframe
        cloudCoordsDf = get_cloud_atom_coords(cloudAtomsDict=cloudAtomsDict,
                                                cofactorName=cofactorName,
                                                pdbDf=pdbDf)
        # generate orb.X count features
        orbCountDf = orb_count(orbCenter=orbCenter,
                                pdbDf=pdbDf,
                                aminoAcidNames=aminoAcidNames,
                                orbValue=orbValue,
                                proteinName=id, 
                                cofactorName=cofactorName)
 
        # generate cloud.X count features
        cloudCountDf = cloud_count(cloudCoordsDf=cloudCoordsDf,
                                    pdbDf=pdbDf,
                                    aminoAcidNames=aminoAcidNames,
                                    cloudValue=cloudValue,
                                    proteinName=id)
        # generate protein.X count features
        proteinCountDf = protein_count(pdbDf=pdbDf,
                                        aminoAcidNames=aminoAcidNames,
                                        proteinName=id)

        # fill in amino acid properties into dataframe using amino acid counts
        propertiesFeaturesDf = cloud_orb_protein_properties(orbCountDf=orbCountDf,
                                                            cloudCountDf=cloudCountDf,
                                                            proteinCountDf=proteinCountDf,
                                                            aminoAcidProperties=aminoAcidProperties,
                                                            aminoAcidNames = aminoAcidNames, 
                                                            proteinName=id)
        # fill in amino acid properties near to key atoms
        keyAtomFeaturesDf = get_key_atom_features(keyAtomCoords=keyAtomCoords,
                                                    pdbDf=pdbDf,
                                                    aminoAcidNames=aminoAcidNames,
                                                    aminoAcidProperties=aminoAcidProperties,
                                                    proteinName=id,
                                                    cofactorName=cofactorName)
        # combine features dataframes and 
        featuresConcat = [orbCountDf,cloudCountDf,proteinCountDf,propertiesFeaturesDf,keyAtomFeaturesDf]
   #     featuresNames = ["orbCountDf","cloudCountDf","proteinCountDf","propertiesFeaturesDf","keyAtomFeaturesDf"]


        featuresDf=pd.concat(featuresConcat,axis=1)
        featuresList.append(featuresDf)
    #    featuresList = [df.reset_index(drop=True) for df in featuresList]

    allFeaturesDf=pd.concat(featuresList,axis=0)
    print(allFeaturesDf)
    allFeaturesDf.to_csv(outputCsv,index=True, sep=",")
    return


########################################################################################
def main():
    # load user inputs
    read_inputs()
    outDir=ifnotmkdir(outputDir)
    # initialise amino acid data
    aminoAcidNames, aminoAcidProperties = initialiseAminoAcidInformation(aminoAcidTable)
    # get list of pdbFiles in pdbDir
    idList, pdbList = getPdbList(inputDir)


    # loop through permutations of orb and cloud values
    jobCount = 0
    totalJobCount = len(orbRange) * len(cloudRange)
    # get a list of job inputs
    jobData = []
    for orbValue in orbRange:
        for cloudValue in cloudRange:
            jobCount += 1
            jobProgress = f"[{str(jobCount)} / {str(totalJobCount)}]"
            jobData.append([orbValue,cloudValue,jobProgress,idList, pdbList, outDir, aminoAcidNames, aminoAcidProperties])
    #run_singlecore(jobData)
    run_multprocessing(jobData)

    print("\nAll features have been generated and saved!")
########################################################################################
def run_singlecore(jobData):
    for jobDetails in jobData:
        process_pdbs(jobDetails)    
########################################################################################
def run_multprocessing(jobData):
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)
    # run with multiprocessing
    pool.map(process_pdbs,jobData)
    pool.close()
    pool.join()           
########################################################################################
if __name__ == "__main__":
    main()