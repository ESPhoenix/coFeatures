
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
