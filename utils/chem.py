import copy
import torch
from io import BytesIO
from openbabel import openbabel
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_scatter import scatter
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit.Chem.rdchem import BondType as BT


BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}



def rdmol_to_data(mol, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    data.nx = to_networkx(data, to_undirected=True)

    return data


def generated_to_xyz(data):
    ptable = Chem.GetPeriodicTable()

    num_atoms = data.ligand_context_element.size(0)
    xyz = "%d\n\n" % (num_atoms, )
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(data.ligand_context_element[i].item())
        x, y, z = data.ligand_context_pos[i].clone().cpu().tolist()
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)

    return xyz


def generated_to_sdf(data):
    xyz = generated_to_xyz(data)
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, xyz)
    sdf = obConversion.WriteString(mol)
    return sdf


def sdf_to_rdmol(sdf):
    stream = BytesIO(sdf.encode())
    suppl = Chem.ForwardSDMolSupplier(stream)
    for mol in suppl:
        return mol
    return None

def generated_to_rdmol(data):
    sdf = generated_to_sdf(data)
    return sdf_to_rdmol(sdf)


def filter_rd_mol(rdmol):
    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]

    # 3-3 ring intersection
    for i, ring_a in enumerate(rings):
        if len(ring_a) != 3:continue
        for j, ring_b in enumerate(rings):
            if i <= j: continue
            inter = ring_a.intersection(ring_b)
            if (len(ring_b) == 3) and (len(inter) > 0): 
                return False

    return True