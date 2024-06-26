
import os
from os import path as p
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

########################################################################################
def normalise_counts_by_size(dataDf, aminoAcidNames, optionsInfo):
    elements = ["C","N","O","S"]
    for region in ["orb","cloud"]:
        ## get totl number of AAs in region
        totalCounts = dataDf[f"{region}.total"]
        # create a list of features to be normalised
        featureList = []
        if optionsInfo["keepIndividualCounts"]:
            for aminoAcid in aminoAcidNames:
                featureList.append(f"{region}.{aminoAcid}")
        if optionsInfo["genAminoAcidCategories"]:
            for category in ["hydrophobic", "aromatic","polar_uncharged","cationic","anionic"]:
                featureList.append(f"{region}.{category}")
        for element in elements:
            featureList.append(f"{region}.{element}")
        ## divide by number of AAs in region
        dataDf[featureList] = dataDf[featureList].div(totalCounts,axis=0)
        ## remove total count features if specified
        if not optionsInfo["keepTotalCounts"]:
            dataDf.drop(columns = [f"{region}.total"], inplace=True)
    return dataDf
###########################################################################################################
def make_amino_acid_category_counts(dataDf, optionsInfo):
    hydrophobicAAs = ["ALA","VAL","ILE","LEU","MET","GLY","PRO"]
    aromaticAAs = ["PHE","TYR","TRP"]
    polarUncharged = ["SER", "THR", "ASN","GLN","HIS","CYS"]
    cationicAAs = ["ARG","LYS"]
    anionicAAs = ["ASP","GLU"]
    aaCategories = {"hydrophobic": hydrophobicAAs,
                    "aromatic": aromaticAAs,
                    "polar_uncharged": polarUncharged,
                    "cationic": cationicAAs,
                    "anionic": anionicAAs}
    
    for region in ["orb","cloud"]:
        for category in aaCategories:
            colNames = [f"{region}.{AA}" for AA in aaCategories[category]]
            dataDf.loc[:,f"{region}.{category}"] = dataDf[colNames].sum(axis=1)
            if not optionsInfo["keepIndividualCounts"]:
                dataDf.drop(columns = colNames, inplace = True)
    return dataDf
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
    idList=[]
    for file in os.listdir(dir):
        fileData = p.splitext(file)
        if fileData[1] == '.pdb':
            idList.append(fileData[0])
    return idList

########################################################################################
def find_cofactor(cofactorInfo,pdbDf):
    # quick check to see what cofactor is present in a pdb file
    # Only works if there is one cofactor - will break with two
    # returns a dictionary of true/false for all cofactors present in confactorNames input variable
    cofactorNames = [key for key in cofactorInfo.keys()]
    cofactorDf = pdbDf[pdbDf["RES_NAME"].isin(cofactorNames)]
    if len(cofactorDf) == 0:
        return 0, 0
    
    uniqueCofactorNames = cofactorDf["RES_NAME"].unique().tolist()
    cofactorCount = 0
    uniqueChains = cofactorDf["CHAIN_ID"].unique().tolist()
    for chainId in uniqueChains:
        chainDf = cofactorDf[cofactorDf["CHAIN_ID"] == chainId]
        uniqueResSeqs = chainDf["RES_SEQ"].unique().tolist()
        cofactorCount += len(uniqueResSeqs)

    
    return cofactorCount , uniqueCofactorNames





########################################################################################
def get_key_atom_coords(pdbDf,keyAtoms,cofactorName):
    cofactorDf = pdbDf[pdbDf["RES_NAME"] == cofactorName]
    ## assumption! different cofactors are stored with different chainIDs!
    uniqueChains = cofactorDf["CHAIN_ID"].unique().tolist()
    keyAtomCoordsList = []
    for chainId in uniqueChains:
        chainDf = cofactorDf[cofactorDf["CHAIN_ID"] == chainId]
        for resId in chainDf["RES_SEQ"].unique().tolist():
            resDf = chainDf[chainDf["RES_SEQ"] == resId]
            keyAtomCoords=pd.DataFrame(columns=["X","Y","Z"], index=keyAtoms)
            for atomId in keyAtoms:
                xCoord = resDf.loc[resDf["ATOM_NAME"] == atomId, "X"].values
                yCoord = resDf.loc[chainDf["ATOM_NAME"] == atomId, "Y"].values
                zCoord = resDf.loc[resDf["ATOM_NAME"] == atomId, "Z"].values
                keyAtomCoords.loc[atomId] = [xCoord,yCoord,zCoord]
            keyAtomCoordsList.append(keyAtomCoords)
    return keyAtomCoordsList
########################################################################################
def gen_orb_region(orbAtoms, cofactorName, pdbDf, orbValue):
    
    ## REMOVE COFACTOR FROM PDB DATAFRAME ##
    noCofactorDf = pdbDf[pdbDf["RES_NAME"] != cofactorName]
    cofactorDf = pdbDf[pdbDf["RES_NAME"] == cofactorName]
    uniqueChains = cofactorDf["CHAIN_ID"].unique().tolist()
    keyAtomCoordsList = []
    orbDfs = []
    for chainId in uniqueChains:
        chainDf = cofactorDf[cofactorDf["CHAIN_ID"] == chainId]

        ## FIND ORB ATOMS IN PDB DATAFRAME ##
        orbDf = chainDf[chainDf["ATOM_NAME"].isin(orbAtoms)]
        # GET ORB CENTER AS AVERAGE POSITION OF ORB ATOMS ##
        orbCenter=[]
        orbCenter.append(orbDf["X"].mean())
        orbCenter.append(orbDf["Y"].mean())
        orbCenter.append(orbDf["Z"].mean())
        
        ## CALCULATE DISTANCES BETWEEN PROTEIN ATOMS AND 
        noCofactorDf.loc[:,"DISTANCES"] = np.linalg.norm(noCofactorDf[["X", "Y", "Z"]].values - np.array(orbCenter), axis=1)
        ## MAKE ORB DATAFRAME OUT OF ATOMS THAT ARE WITHIN ORBVALUE OF ORB CENTER
        orbDf = noCofactorDf[noCofactorDf["DISTANCES"] <= orbValue].copy()
        orbDfs.append(orbDf)
    orbDf = pd.concat(orbDfs, axis=0, ignore_index=True)
    return orbDf
########################################################################################
def gen_cloud_region(cloudAtoms, cofactorName, pdbDf, cloudValue):
    ## FIND CLOUD ATOMS IN PDB DATAFRAME ##
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
    regionDf['ELEMENT'] = regionDf['ATOM_NAME'].apply(lambda x: x[0]).copy()
    elemCount = regionDf["ELEMENT"].value_counts()
    for element in ["C","N","O","S"]:
        try:
            elementCountDf.loc[:,f'{regionName}.{element}'] = elemCount[element]
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

def nearest_n_residues_to_key_atom(keyAtomCoordsList, pdbDf, aminoAcidNames, aminoAcidProperties,
                                   proteinName,cofactorName, nNearestList) :
    ## REMOVE COFACTOR FROM PDB DATAFRAME ##
    noCofactorDf = pdbDf[pdbDf["RES_NAME"] != cofactorName]

    keyAtomFeaturesList = []

    for keyAtomCoords in keyAtomCoordsList:
        ## INITAILSE KEY ATOMS DATAFRAME ##       
        columnNames = []
        for keyAtomId, (x,y,z) in keyAtomCoords.iterrows():
            for property in aminoAcidProperties:
                for nNearest in nNearestList:
                    columnNames.append(f'{str(nNearest)}_{keyAtomId}.{property}')
        keyAtomFeaturesDf = pd.DataFrame(columns=columnNames, index=[proteinName])

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
                        try:
                            value = aminoAcidProperties.at[aminoAcid,property]
                        except:
                            value = 0
                        propertyValue += value 
                    propertyValue = propertyValue / nNearest
                    ## ADD TO DATAFRAME ##
                    keyAtomFeaturesDf.loc[proteinName,f'{str(nNearest)}_{keyAtomId}.{property}'] = propertyValue
        keyAtomFeaturesList.append(keyAtomFeaturesDf)

        keyAtomFeaturesDf = pd.concat(keyAtomFeaturesList).groupby(level=0).sum()

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
