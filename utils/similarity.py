import numpy as np
from rdkit import Chem, DataStructs


def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    return DataStructs.TanimotoSimilarity(fp1,fp2)
    

def tanimoto_sim_N_to_1(mols, ref):
    sim = [tanimoto_sim(m, ref) for m in mols]
    return sim


def batched_number_of_rings(mols):
    n = []
    for m in mols:
        n.append(Chem.rdMolDescriptors.CalcNumRings(m))
    return np.array(n)
