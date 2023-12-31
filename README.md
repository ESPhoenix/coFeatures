# coFeatures
Cofactor-dependent feature generation for protiens
 
 ## Clone Repository
 ```bash
 git clone https://github.com/ESPhoenix/coFeatures
 ```

 ## Create and Activate Conda Environment
 ```bash
 conda create -n coFeaturesEnv python=3.7.12
 conda activate coFeaturesEnv
 ```

 ## Install Required Python Packages
 ```bash
pip install argpass==0.0.2 numpy==1.21.6 pandas==1.3.5 tqdm==4.66.1
 ```

 ## Edit Config File or Create Your Own
 ```python
import numpy as np        
def inputs():
        ## LOCATION OF YOUR PDB FILES
        inputDir = "/home/esp/featureGenerationMethods/coFeatures/test_pdbs"       
        ## WHERE DO YOU WANT YOUR OUTPUTS
        outDir = "/home/esp/featureGenerationMethods/coFeatures/features"          
        ## LOCATION OF AA PROPERTIES
        aminoAcidTable="/home/esp/featureGenerationMethods/coFeatures/amino_acid_properties.txt" 
        # cofactor information
        ## LIST OF COFACTOR NAMES IN YOUR PDB FILES
        cofactorNames = ["FAD", "FMN"]
        ## DICT CONTAINING: RESIDUE_NAME:[ATOM1,ATOM2...]
        ## WILL BE USED TO GENERATE NEAREST 1 and NEAREST 3 RESIDUES FEATURES
        keyAtomsDict = {
                "FMN":["N1","N3","N5"],
                "FAD":["N1","N3","N5"]
                }
        ## DICT CONTAINING: RESIDUE_NAME:[ATOM1,ATOM2...]
        ## WILL BE USED TO DEFINE CENTER OF ORB REGION 
        orbAtomsDict = {
                "FMN":["N10","C10","C4A","N5","C5A","C9A"],
                "FAD":["N10","C10","C4X","N5","C5X","C9A"]
                        }
        ## DICT CONTAINING: RESIDUE_NAME:[ATOM1,ATOM2...]
        ## WILL BE USED TO DEFINE ATOMS TO CONTRUCT CLOUD REGION                 
        cloudAtomsDict = {
                "FMN":["N10","C9A","C9","C8","C8M","C7","C7M","C6","C5A","N5",
                        "C4A","C4","O4","N3","HN3","C2","O2","N1","C10"],
                "FAD":["N10","C9A","C9","C8","C8M","C7","C7M","C6","C5X","N5",
                        "C4X","C4","O4","N3","HN3","C2","O2","N1","C10"]
                        }
        ## LIST OF RADII FOR ORB REGION
        orbRange = list(np.arange(6, 10))
        ## LIST OF RADII FOR CLOUD REGION
        cloudRange = list(np.arange(3, 7))                      

        return (inputDir, outDir, aminoAcidTable, cofactorNames, keyAtomsDict,
                    orbAtomsDict, cloudAtomsDict, orbRange, cloudRange)

 ```

 ## Useage
 From the coFeatures directory, run the following in the command line:
 ```bash
 python coFeatures.py --config config.py
 ```
You may want to change the name of config.py to your own config file
