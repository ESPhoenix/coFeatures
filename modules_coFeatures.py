
import os
from os import path as p
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
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
    return idList

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
       # print(f"no cofactor found in {protName}")
        cofactorCountWrong = True
        return None, cofactorCountWrong
    elif countTrue > 1:
        #print(f'{countTrue} cofactors found in {protName}, need only one!')
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
def gen_orb_region(orbAtomsDict, cofactorName, pdbDf, orbValue):
    ## FIND ORB ATOMS IN PDB DATAFRAME ##
    orbAtoms = orbAtomsDict.get(cofactorName,[])
    cofactorRows = pdbDf[pdbDf["RES_NAME"] == cofactorName]
    orbRows = cofactorRows[cofactorRows["ATOM_NAME"].isin(orbAtoms)]
    # GET ORB CENTER AS AVERAGE POSITION OF ORB ATOMS ##
    orbCenter=[]
    orbCenter.append(orbRows["X"].mean())
    orbCenter.append(orbRows["Y"].mean())
    orbCenter.append(orbRows["Z"].mean())

    ## REMOVE COFACTOR FROM PDB DATAFRAME ##
    noCofactorDf = pdbDf[pdbDf["RES_NAME"] != cofactorName]
    
    ## CALCULATE DISTANCES BETWEEN PROTEIN ATOMS AND 
    noCofactorDf.loc[:,"DISTANCES"] = np.linalg.norm(noCofactorDf[["X", "Y", "Z"]].values - np.array(orbCenter), axis=1)
    ## MAKE ORB DATAFRAME OUT OF ATOMS THAT ARE WITHIN ORBVALUE OF ORB CENTER
    orbDf = noCofactorDf[noCofactorDf["DISTANCES"] <= orbValue].copy()

    return orbDf
########################################################################################
def gen_cloud_region(cloudAtomsDict, cofactorName, pdbDf, cloudValue):
    ## FIND CLOUD ATOMS IN PDB DATAFRAME ##
    cloudAtoms = cloudAtomsDict.get(cofactorName,[])
    cofactorRows = pdbDf[pdbDf["RES_NAME"] == cofactorName]
    cloudRows = cofactorRows[cofactorRows["ATOM_NAME"].isin(cloudAtoms)]

    ## CREATE DATAFRAME CONTAINING COORDINATES OF CLOUD ATOMS ##
    cloudCoordsDf=cloudRows[["ATOM_NAME","X","Y","Z"]]
    cloudCoordsDf.set_index("ATOM_NAME",inplace=True)

    ## REMOVE COFACTOR FROM PDB DATAFRAME ##
    noCofactorDf = pdbDf[pdbDf["RES_NAME"] != cofactorName]

    ## LOOP THROUGH CLOUD ATOMS ##
    for i, (x,y,z) in cloudCoordsDf.iterrows():
        ## CALCULATE DISTANCE BETWEEN CLOUD ATOMS AND PROTEIN ATOMS ##
        columnName=f'cloud_dist_atom_{i}'
        noCofactorDf.loc[:,columnName] = np.linalg.norm(noCofactorDf[["X", "Y", "Z"]].values - np.array([x,y,z]), axis=1)

    ## SEPARATE CLOUD DISTANCE COLUMNS ##
    cloudCols = [col for col in noCofactorDf.columns if col.startswith('cloud_dist_atom_')]
    ## SELECT ATOMS IN PROTEIN DATAFRAME WITHIN CLOUDVALUE OF ANY CLOUD ATOM ##
    withincloudRange = noCofactorDf[cloudCols].lt(cloudValue).any(axis=1)
    cloudDf=noCofactorDf[withincloudRange]

    return cloudDf
        
#######################################################################################
def gen_protein_region(pdbDf,cofactorName):
    ## REMOVE COFACTOR FROM PDB DATAFRAME ##
    proteinDf = pdbDf[pdbDf["RES_NAME"] != cofactorName]
    return proteinDf

########################################################################################
def element_count_in_region(regionDf,regionName,proteinName):
    ## INITIALISE ELEMENT COUNT DATAFRAME ##
    columnNames=[]
    for element in ["C","N","O","S"]:
        columnNames.append(f"{regionName}.{element}")
    elementCountDf = pd.DataFrame(columns=columnNames,index=[proteinName])
    ## COUNT ELEMENTS IN REGION, RETURN ZERO IF REGION HAS NONE OR DOES NOT EXIST
    for element in ["C","N","O","S"]:
        try:
            elementCountDf.loc[:,f'{regionName}.{element}'] = regionDf["ELEMENT"].value_counts()[element]
        except:
            elementCountDf.loc[:, f'{regionName}.{element}'] = 0

    return elementCountDf
########################################################################################
def amino_acid_count_in_region(regionDf, regionName, proteinName, aminoAcidNames):
    ## INITIALSE AMINO ACID COUNT DATAFRAME ##
    columnNames=[]
    for aminoAcid in aminoAcidNames:
        columnNames.append(f"{regionName}.{aminoAcid}")
    aaCountDf = pd.DataFrame(columns=columnNames,index=[proteinName])

    ## GET UNIQUE RESIDUES ONLY ##
    uniqueResiduesDf = regionDf.drop_duplicates(subset = ["RES_SEQ"])

    ## COUNT AMINO ACIDS IN REGION, RETURN ZERO IF REGION HAS NONE OR DOES NOT EXIST
    totalResidueCount = []
    for aminoAcid in aminoAcidNames:
        try: 
            aaCountDf.loc[:,f'{regionName}.{aminoAcid}'] = uniqueResiduesDf["RES_NAME"].value_counts()[aminoAcid]
        except:
            aaCountDf.loc[:,f'{regionName}.{aminoAcid}'] = 0

    aaCountDf.loc[:,f"{regionName}.total"] = aaCountDf.sum(axis=1)
    return aaCountDf
########################################################################################
def calculate_amino_acid_properties_in_region(aaCountDf, aminoAcidNames, aminoAcidProperties, proteinName, regionName):
    ## INITIALISE PROPERTIES DATAFRAME ##
    columnNames = []
    for property in aminoAcidProperties.columns:
        columnNames.append(f"{regionName}.{property}")
    propertiesDf = pd.DataFrame(columns=columnNames, index=[proteinName])
    
    ## LOOP THROUGH PROPERTIES SUPPLIED IN AMINO_ACID_TABLE.txt
    for property in aminoAcidProperties:
        propertyValue=0
        for aminoAcid in aminoAcidNames:
            try:
                aaCount = aaCountDf.at[proteinName,f"{regionName}.{aminoAcid}"]
            except KeyError:
                aaCount = 0
            aaPropertyvalue = aminoAcidProperties.at[aminoAcid,property]
            value = aaCount * aaPropertyvalue
            propertyValue += value 
        try:
            totalAminoAcids=aaCountDf.at[proteinName,f'{regionName}.total']
        except KeyError:
            totalAminoAcids=0
        if not totalAminoAcids == 0:
            propertyValue = propertyValue / totalAminoAcids
        propertiesDf[f'{regionName}.{property}'] = propertyValue

    return propertiesDf
########################################################################################

def nearest_n_residues_to_key_atom(keyAtomCoords, pdbDf, aminoAcidNames, aminoAcidProperties,
                                   proteinName,cofactorName, nNearestList) :
    ## INITAILSE KEY ATOMS DATAFRAME ##       
    columnNames = []
    for keyAtomId, (x,y,z) in keyAtomCoords.iterrows():
        for property in aminoAcidProperties:
            for nNearest in nNearestList:
                columnNames.append(f'{str(nNearest)}_{keyAtomId}.{property}')
    keyAtomFeaturesDf = pd.DataFrame(columns=columnNames, index=[proteinName])


    ## REMOVE COFACTOR FROM PDB DATAFRAME ##
    noCofactorDf = pdbDf[pdbDf["RES_NAME"] != cofactorName]
    ## LOOP THROUGH KEY ATOMS ##
    for keyAtomId, (x,y,z) in keyAtomCoords.iterrows():
        coordsReshaped = np.array([x,y,z]).reshape(1,-1)
        ## CALCULATE DISTANCE TO KEY ATOM FOR ALL PROTEIN ATOMS ##
        columnName=f'{keyAtomId}_distance'
        noCofactorDf.loc[:,columnName] = np.linalg.norm((noCofactorDf[["X", "Y", "Z"]].values - coordsReshaped), axis=1)
        ## GET LIST OF UNIQUE RESIDUE NUMBERS ##
        uniqueResidues = noCofactorDf['RES_SEQ'].unique()

        ## FIND NEAREST ATOM IN EACH RESIDUE TO KEY ATOM ##
        nearestPointIndecies=[]
        for residueSeq in uniqueResidues:
            residueData = noCofactorDf[noCofactorDf['RES_SEQ'] == residueSeq]
            nearestPointIdx = residueData[columnName].idxmin()
            nearestPointIndecies.append(nearestPointIdx)    

        ## GET NEW DATAFRAME CONTAINING ONLY NEAREST ATOMS OF EACH RESIDUE ##
        nearestPointsResidues = noCofactorDf.loc[nearestPointIndecies]
        ## RESET INDEX ##
        nearestPointsResidues.reset_index(drop=True, inplace=True)
        ## SORT BY DISTANCE TO KEY ATOM ##
        nearestResiduesSorted = nearestPointsResidues.sort_values(by=columnName, ascending=True)
        ## LOOP THROUGH NNEARESTLIST ##
        for nNearest in nNearestList:
            ## GET LIST OF N NEAREST RESIDUES TO KEY ATOM ##
            nearestResidueList = nearestResiduesSorted.head(nNearest)["RES_NAME"].to_list()
            ## GET PROPERTY VALUES ##
            for property in aminoAcidProperties:
                propertyValue=0
                for aminoAcid in nearestResidueList:
                    value = aminoAcidProperties.at[aminoAcid,property]
                    propertyValue += value 
                propertyValue = propertyValue / nNearest
                ## ADD TO DATAFRAME ##
                keyAtomFeaturesDf.loc[proteinName,f'{str(nNearest)}_{keyAtomId}.{property}'] = propertyValue

    return keyAtomFeaturesDf
########################################################################################
def merge_temporary_csvs(outDir,orbRange,cloudRange):
    for orbValue in orbRange:
        for cloudValue in cloudRange:
            dfsToConcat = []
            for file in os.listdir(outDir):
                if not p.splitext(file)[1] == ".csv":
                    continue
                if f"{str(orbValue)}_{str(cloudValue)}" in file:
                    df = pd.read_csv(p.join(outDir,file))
                    dfsToConcat.append(df)
                    os.remove(p.join(outDir,file))
            featuresDf = pd.concat(dfsToConcat,axis=0)
            saveFile = p.join(outDir,f"coFeatures_orb_{str(orbValue)}_cloud_{str(cloudValue)}.csv")
            featuresDf.to_csv(saveFile,index=False, sep=",")
