########################################################################################
## BASIC LIBRARIES ##
import os
from os import path as p
import sys
import pandas as pd
import multiprocessing as mp
import argpass
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
    os.makedirs(outDir)
    # READ AMINO ACID TABLE INTO A DATAFRAME, GET LIST OF AMIO ACID NAMES ##
    aminoAcidNames, aminoAcidProperties = initialiseAminoAcidInformation(aminoAcidTable)

    # GET LISTS OF PDB IDS AND PATHS
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