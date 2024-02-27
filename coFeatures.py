########################################################################################
## BASIC LIBRARIES ##
import os
from os import path as p
import sys
import pandas as pd
import multiprocessing
import argpass
from itertools import product
import multiprocessing
from tqdm import tqdm
import yaml
## coFeatures SPESIFIC MODULES ##
from utils_coFeatures import *
from modules_coFeatures import *
########################################################################################
def read_inputs():
    ## create an argpass parser, read config file, snip off ".py" if on the end of file
    parser = argpass.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config=args.config
    ## Read config.yaml into a dictionary
    with open(config,"r") as yamlFile:
        config = yaml.safe_load(yamlFile) 

    pathInfo = config["pathInfo"]
    cofactorInfo = config["cofactorInfo"]
    optionsInfo = config["optionsInfo"]

    return pathInfo,cofactorInfo, optionsInfo
########################################################################################
def main():
    ## LOAD USER INPUTS ##
    pathInfo,cofactorInfo, optionsInfo = read_inputs()
    # MAKE OUTPUT DIRECTORY ##
    os.makedirs(pathInfo["outDir"],exist_ok=True)
    # READ AMINO ACID TABLE INTO A DATAFRAME, GET LIST OF AMIO ACID NAMES ##
    # GET LISTS OF PDB IDS AND PATHS
    idList = getPdbList(pathInfo["inputDir"])

    jobOrder = list(product(idList,optionsInfo["orbRange"],optionsInfo["cloudRange"]))

    # process_pdbs_multicore(jobOrder, pathInfo, cofactorInfo)

    
    process_pdbs_singlecore(jobOrder, pathInfo, cofactorInfo)
    
    merge_temporary_csvs(outDir = pathInfo["outDir"],
                        orbRange = optionsInfo["orbRange"],
                        cloudRange = optionsInfo["cloudRange"])

 
    print("\nAll features have been generated and saved!")
########################################################################################
def  process_pdbs_singlecore(jobOrder, pathInfo, cofactorInfo):
    
    aminoAcidNames, aminoAcidProperties = initialiseAminoAcidInformation(pathInfo["aminoAcidTable"])

    for jobDetails in jobOrder:
        process_pdbs_worker(jobDetails, pathInfo, cofactorInfo, aminoAcidNames, aminoAcidProperties)
########################################################################################
def process_pdbs_multicore(jobOrder, pathInfo, cofactorInfo):
    
    aminoAcidNames, aminoAcidProperties = initialiseAminoAcidInformation(pathInfo["aminoAcidTable"])
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_pdbs_worker,
                     tqdm( [(jobDetails, pathInfo, cofactorInfo, aminoAcidNames, aminoAcidProperties) for jobDetails in jobOrder],
                            total = len(jobOrder)))

########################################################################################

def process_pdbs_worker(jobDetails, pathInfo, cofactorInfo, aminoAcidNames, aminoAcidProperties):
    ## UNPACK pathInfo
    pdbDir = pathInfo["inputDir"]
    outDir = pathInfo["outDir"]


    ## UNPACK JOB DETAILS INTO VARIABLES ##
    pdbId, orbValue, cloudValue = jobDetails
    pdbFile = p.join(pdbDir,f"{pdbId}.pdb")

    ## INITIALSE TEMPORARY OUTPUT FILE, SKIP IF IT ALREADY EXISTS ##
    outputCsv=p.join(outDir, f"{pdbId}_{str(orbValue)}_{str(cloudValue)}.tmp")
    if p.isfile(outputCsv):
        return
    # INITIALISE LIST TO STORE ALL FEATURE DATAFRAMES ##
    pdbDf=pdb2df(pdbFile)
    cofactorName, cofactorCountWrong = find_cofactor(pdbDf=pdbDf,cofactorInfo=cofactorInfo)
    ## SKIP IF MORE THAN ONE OR ZERO COFACTORS PRESENT ##
    if cofactorCountWrong:
        return

    ## GET ORB, CLOUD, AND PROTEIN REGION DATAFRAMES ##
    orbDf = gen_orb_region(orbAtoms=cofactorInfo[cofactorName]["orbAtoms"],
                           cofactorName=cofactorName,
                           pdbDf=pdbDf,
                           orbValue=orbValue)
    cloudDf = gen_cloud_region(cloudAtoms=cofactorInfo[cofactorName]["cloudAtoms"],
                           cofactorName=cofactorName,
                           pdbDf=pdbDf,
                           cloudValue=cloudValue)
    proteinDf = gen_protein_region(pdbDf=pdbDf,
                                   cofactorName=cofactorName)
    
    ## COUNT ELEMENTS IN REGIONS ##
    orbElementCountDf       = element_count_in_region(regionDf=orbDf,
                                                        regionName="orb",
                                                        proteinName=pdbId)
    cloudElementCountDf     = element_count_in_region(regionDf=cloudDf,
                                                        regionName="cloud",
                                                        proteinName=pdbId)
    proteinElementCountDf   = element_count_in_region(regionDf=proteinDf,
                                                        regionName="protein",
                                                        proteinName=pdbId)
    
    ## COUNT AMINO ACIDS IN REGIONS ##
    orbAACountDf            = amino_acid_count_in_region(regionDf=orbDf,
                                              regionName="orb",
                                              proteinName=pdbId,
                                              aminoAcidNames=aminoAcidNames)
    cloudAACountDf          = amino_acid_count_in_region(regionDf=cloudDf,
                                                regionName="cloud",
                                                proteinName=pdbId,
                                                aminoAcidNames=aminoAcidNames)
    proteinAACountDf        = amino_acid_count_in_region(regionDf=proteinDf,
                                                  regionName="protein",
                                                  proteinName=pdbId,
                                                  aminoAcidNames=aminoAcidNames)
    ## AMINO ACID PROPERTIES FOR REGIONS ##
    orbPropertiesDf         = calculate_amino_acid_properties_in_region(aaCountDf=orbAACountDf,
                                                                        aminoAcidProperties=aminoAcidProperties,
                                                                        aminoAcidNames=aminoAcidNames,
                                                                        proteinName=pdbId,
                                                                        regionName="orb")
    cloudPropertiesDf       = calculate_amino_acid_properties_in_region(aaCountDf=cloudAACountDf,
                                                                        aminoAcidProperties=aminoAcidProperties,
                                                                        aminoAcidNames=aminoAcidNames,
                                                                        proteinName=pdbId,
                                                                        regionName="cloud")
    proteinPropertiesDf     = calculate_amino_acid_properties_in_region(aaCountDf=proteinAACountDf,
                                                                        aminoAcidProperties=aminoAcidProperties,
                                                                        aminoAcidNames=aminoAcidNames,
                                                                        proteinName=pdbId,
                                                                        regionName="protein")
    ## EXTRACT COORDINATES OF USER-DEFINED KEY ATOMS ##
    keyAtomCoords = get_key_atom_coords(pdbDf = pdbDf,
                                        keyAtoms = cofactorInfo[cofactorName]["keyAtoms"],
                                        cofactorName = cofactorName) 
    ## NEAREST AMINO ACIDS TO KEY ATOMS ##
    keyAtomsFeaturesDf        = nearest_n_residues_to_key_atom(keyAtomCoords=keyAtomCoords,
                                                             pdbDf=pdbDf,
                                                             aminoAcidNames=aminoAcidNames,
                                                             aminoAcidProperties=aminoAcidProperties,
                                                             proteinName=pdbId,
                                                             cofactorName=cofactorName,
                                                             nNearestList=[1,3])
    # combine features dataframes and 
    featuresToConcat = [orbElementCountDf, cloudElementCountDf, proteinElementCountDf,
                        orbAACountDf, cloudAACountDf, proteinAACountDf,
                        orbPropertiesDf, cloudPropertiesDf, proteinPropertiesDf,
                        keyAtomsFeaturesDf]


    featuresDf=pd.concat(featuresToConcat,axis=1)

    tmpSaveFile = p.join(outDir,f"{pdbId}_{str(orbValue)}_{str(cloudValue)}.csv")
    featuresDf.to_csv(tmpSaveFile,index=True, sep=",")
    return
########################################################################################
if __name__ == "__main__":
    main()