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
## coFeatures SPESIFIC MODULES ##
from utils_coFeatures import *
from modules_coFeatures import *
########################################################################################
def read_inputs():
    ## MAKE A PARSER TO READ --CONFIG FLAG ##
    parser = argpass.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    configName=args.config
    ## DEAL WITH .PY EXTENSION IF PRESENT ##
    if  args.config == None:
        print('No config file name provided.')
    configName = p.splitext(configName)[0]

    ## ADD CONFIG FILE TO PYTHONPATH ##
    cwd = os.getcwd()
    configPath = p.join(cwd,configName)
    sys.path.append(configPath)
    ## IMPORT CONFIG FILE AND RETURN VARIABLES CONTAINED ##
    try:
        config_module = __import__(configName)
        (inputDir, outDir, aminoAcidTable, cofactorNames, keyAtomsDict,
          orbAtomsDict, cloudAtomsDict, orbRange, cloudRange)= config_module.inputs()
        
        return (inputDir, outDir, aminoAcidTable, cofactorNames, keyAtomsDict,
                orbAtomsDict, cloudAtomsDict, orbRange, cloudRange)
    except ImportError:
        print(f"Error: Can't  import module '{configName}'. Make sure the input exists!")
        print("HOPE IS THE FIRST STEP ON THE ROAD TO DISAPPOINTMENT")
        exit()
########################################################################################
def main():
    ## LOAD USER INPUTS ##
    (inputDir, outDir, aminoAcidTable, cofactorNames, keyAtomsDict,
                orbAtomsDict, cloudAtomsDict, orbRange, cloudRange) = read_inputs()
    # MAKE OUTPUT DIRECTORY ##
    os.makedirs(outDir,exist_ok=True)
    # READ AMINO ACID TABLE INTO A DATAFRAME, GET LIST OF AMIO ACID NAMES ##
    aminoAcidNames, aminoAcidProperties = initialiseAminoAcidInformation(aminoAcidTable)
    # GET LISTS OF PDB IDS AND PATHS
    idList = getPdbList(inputDir)

    jobOrder = list(product(idList,orbRange,cloudRange))


    process_pdbs_multicore(jobOrder = jobOrder,
                outDir =   outDir,
                aminoAcidNames = aminoAcidNames,
                aminoAcidProperties = aminoAcidProperties,
                pdbDir = inputDir,
                cofactorNames = cofactorNames,
                keyAtomsDict = keyAtomsDict,
                orbAtomsDict = orbAtomsDict,
                cloudAtomsDict = cloudAtomsDict)
    
    #process_pdbs_singlecore(jobOrder = jobOrder,
    #             outDir =   outDir,
    #             aminoAcidNames = aminoAcidNames,
    #             aminoAcidProperties = aminoAcidProperties,
    #             pdbDir = inputDir,
    #             cofactorNames = cofactorNames,
    #             keyAtomsDict = keyAtomsDict,
    #             orbAtomsDict = orbAtomsDict,
    #             cloudAtomsDict = cloudAtomsDict)
    
    merge_temporary_csvs(outDir = outDir,
                        orbRange = orbRange,
                        cloudRange = cloudRange)

 
    print("\nAll features have been generated and saved!")
########################################################################################
def  process_pdbs_singlecore(jobOrder, outDir, aminoAcidNames, aminoAcidProperties,
                  pdbDir,cofactorNames, keyAtomsDict,orbAtomsDict, cloudAtomsDict):
    for jobDetails in jobOrder:
        process_pdbs_worker(jobDetails, outDir, aminoAcidNames, aminoAcidProperties,
                              pdbDir,cofactorNames, keyAtomsDict,orbAtomsDict, cloudAtomsDict)
########################################################################################
def process_pdbs_multicore(jobOrder, outDir, aminoAcidNames, aminoAcidProperties,
                  pdbDir,cofactorNames, keyAtomsDict,orbAtomsDict, cloudAtomsDict):
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_pdbs_worker,
                     tqdm( [(jobDetails, outDir, aminoAcidNames, aminoAcidProperties,
                              pdbDir,cofactorNames, keyAtomsDict,orbAtomsDict, cloudAtomsDict) for jobDetails in jobOrder],
                            total = len(jobOrder)))

########################################################################################

def process_pdbs_worker(jobDetails, outDir, aminoAcidNames, aminoAcidProperties,
                        pdbDir,cofactorNames, keyAtomsDict,orbAtomsDict, cloudAtomsDict):
    ## UNPACK JOB DETAILS INTO VARIABLES ##
    pdbId, orbValue, cloudValue = jobDetails
    pdbFile = p.join(pdbDir,f"{pdbId}.pdb")

    ## INITIALSE TEMPORARY OUTPUT FILE, SKIP IF IT ALREADY EXISTS ##
    outputCsv=p.join(outDir, f"{pdbId}_{str(orbValue)}_{str(cloudValue)}.tmp")
    if p.isfile(outputCsv):
        return
    # INITIALISE LIST TO STORE ALL FEATURE DATAFRAMES ##
    pdbDf=pdb2df(pdbFile)
    cofactorName, cofactorCountWrong = find_cofactor(pdbDf=pdbDf,
                                                          cofactorNames=cofactorNames, 
                                                          protName=pdbId)
    ## SKIP IF MORE THAN ONE OR ZERO COFACTORS PRESENT ##
    if cofactorCountWrong:
        return

    ## GET ORB, CLOUD, AND PROTEIN REGION DATAFRAMES ##
    orbDf = gen_orb_region(orbAtomsDict=orbAtomsDict,
                           cofactorName=cofactorName,
                           pdbDf=pdbDf,
                           orbValue=orbValue)
    cloudDf = gen_cloud_region(cloudAtomsDict=cloudAtomsDict,
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
    keyAtomCoords = get_key_atom_coords(pdbDf,keyAtomsDict,cofactorName) 
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