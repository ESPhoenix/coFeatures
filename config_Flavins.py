import numpy as np        
def inputs():
        # directory information
        inputDir = "/home/eugene/FeatureGeneration/coFeatures_inputs"
        outDir = "/home/eugene/FeatureGeneration/coFeatures_features"
        aminoAcidTable="/home/eugene/FeatureGeneration/tableAmm.txt"
        # cofactor information
        cofactorNames = ["FAD", "FMN"]
        keyAtomsDict = {
                "FMN":["N1","N3","N5"],
                "FAD":["N1","N3","N5"]
                }
        orbAtomsDict = {
                "FMN":["N10","C10","C4A","N5","C5A","C9A"],
                "FAD":["N10","C10","C4X","N5","C5X","C9A"]
                        }
        cloudAtomsDict = {
                "FMN":["N10","C9A","C9","C8","C8M","C7","C7M","C6","C5A","N5",
                        "C4A","C4","O4","N3","HN3","C2","O2","N1","C10"],
                "FAD":["N10","C9A","C9","C8","C8M","C7","C7M","C6","C5X","N5",
                        "C4X","C4","O4","N3","HN3","C2","O2","N1","C10"]
                        }
        orbRange = list(np.arange(6, 10))
        cloudRange = list(np.arange(3, 7))                      

        return inputDir, outDir, aminoAcidTable, cofactorNames, keyAtomsDict, orbAtomsDict, cloudAtomsDict, orbRange, cloudRange


