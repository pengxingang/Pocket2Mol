import os
import subprocess
import random
import string
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit.Chem import AllChem
from utils.reconstruct import reconstruct_from_generated_with_edges
from rdkit.Chem.rdMolAlign import CalcRMS
import numpy as np

def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length)) 


def load_pdb(path):
    with open(path, 'r') as f:
        return f.read()


def parse_qvina_outputs(docked_sdf_path, ref_mol):

    suppl = Chem.SDMolSupplier(docked_sdf_path)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        try:
            rmsd = CalcRMS(ref_mol, mol)
        except:
            rmsd = np.nan
        results.append(EasyDict({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
            'rmsd_ref': rmsd
        }))

    return results

class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()
    
    def get_results(self):
        raise NotImplementedError()


class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_specific_data(cls, mol, pdb_block, **kwargs):
        return cls(pdb_block, mol, **kwargs)

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
        ligand_rdmol = reconstruct_from_generated_with_edges(data)
        return cls(pdb_block, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked', **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(pdb_block, ligand_rdmol, **kwargs)

    def __init__(self, pdb_block, ligand_rdmol, conda_env='adt', tmp_dir='./tmp', use_uff=True, center=None):
        super().__init__(pdb_block, ligand_rdmol)
        self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)

        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        if use_uff:
            try:
                not_converge = 10
                while not_converge > 0:
                    flag = UFFOptimizeMolecule(ligand_rdmol)
                    not_converge = min(not_converge - 1, flag * 10)
            except RuntimeError:
                pass
        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol
        self.noH_rdmol = Chem.RemoveHs(ligand_rdmol)

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        self.proc = None
        self.results = None
        self.output = None
        self.docked_sdf_path = None

    def run(self, exhaustiveness=16):
        commands = """
eval "$(conda shell.bash hook)"
conda activate {env}
cd {tmp}
# Prepare receptor (PDB->PDBQT)
prepare_receptor4.py -r {receptor_id}.pdb
# Prepare ligand
obabel {ligand_id}.sdf -O{ligand_id}.pdbqt
qvina2 \
    --receptor {receptor_id}.pdbqt \
    --ligand {ligand_id}.pdbqt \
    --center_x {center_x:.4f} \
    --center_y {center_y:.4f} \
    --center_z {center_z:.4f} \
    --size_x 20 --size_y 20 --size_z 20 \
    --exhaustiveness {exhaust}
obabel {ligand_id}_out.pdbqt -O{ligand_id}_out.sdf -h
        """.format(
            receptor_id = self.receptor_id,
            ligand_id = self.ligand_id,
            env = self.conda_env, 
            tmp = self.tmp_dir, 
            exhaust = exhaustiveness,
            center_x = self.center[0],
            center_y = self.center[1],
            center_z = self.center[2],
        )

        self.docked_sdf_path = os.path.join(self.tmp_dir, '%s_out.sdf' % self.ligand_id)

        self.proc = subprocess.Popen(
            '/bin/bash', 
            shell=False, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )

        self.proc.stdin.write(commands.encode('utf-8'))
        self.proc.stdin.close()

        # return commands

    def run_sync(self):
        self.run()
        while self.get_results() is None:
            pass
        results = self.get_results()
        print('Best affinity:', results[0]['affinity']) #, 'RMSD:', results[0]['rmsd_ref'])
        return results

    def get_results(self):
        if self.proc is None:   # Not started
            return None
        elif self.proc.poll() is None:  # In progress
            return None
        else:
            if self.output is None:
                self.output = self.proc.stdout.readlines()
                try:
                    self.results = parse_qvina_outputs(self.docked_sdf_path, self.noH_rdmol)
                except:
                    print('[Error] Vina output error: %s' % self.docked_sdf_path)
                    return []
            return self.results

