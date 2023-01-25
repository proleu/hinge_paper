##### thank you phil!!!

from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [options] FILE", version="0.1")
parser.add_option("--pdb", type="string", dest="pdb", metavar="STR", help="Path to pdb")
parser.add_option("--outdir", type="string", dest="outdir", metavar="STR", help="Name of output folder")
(opts, args) = parser.parse_args()
parser.set_defaults()

import sys
sys.path.append("/projects/crispy_shifty/")
from crispy_shifty.protocols import mpnn
import pyrosetta
from crispy_shifty.protocols.mpnn import MPNNDesign
from pyrosetta.rosetta.core.select.residue_selector import (
    AndResidueSelector,
    ChainSelector,
    NeighborhoodResidueSelector,
    NotResidueSelector,
    ResidueIndexSelector,
    OrResidueSelector,
)

pdb = opts.pdb
pdbname = opts.pdb.split('/')[-1].replace('.pdb', '')

# TODO: init with the ligand in the pdb
with open(pdb, 'r') as f:
	pocket_resi = [line.split()[2] for line in f if ('REMARK' in line and 'pocket' in line)]
print(pocket_resi)
with open(pdb, 'r') as f:
	ligand_name = [line.split()[5] for line in f if 'REMARK 666' in line][0]

pyrosetta.init(
    f"-extra_res_fa /home/laukoa/Projects/serine_hydrolase/theozyme/{ligand_name}/{ligand_name}.params"
)
pose = pyrosetta.pose_from_file(pdb)

chA_selector = ChainSelector("A")
ligand = ResidueIndexSelector()
ligand.set_index(pose.size())

pocket = ResidueIndexSelector()
pocket.set_index(','.join(pocket_resi))
surface = NotResidueSelector(pocket)

neighborhood = NeighborhoodResidueSelector()
neighborhood.set_distance(12.0)
neighborhood.set_focus_selector(ligand)

not_neighborhood = NotResidueSelector(neighborhood)
A_not_neighborhood = AndResidueSelector(chA_selector, not_neighborhood)

surface_or_not_neighborhood = OrResidueSelector(surface, A_not_neighborhood)

allowed_indices = [
    str(i)
    for i, allowed in enumerate(list(surface_or_not_neighborhood.apply(pose)), start=1)
    if (allowed and i <= pose.chain_end(1))
]
design_selector = ResidueIndexSelector(",".join(allowed_indices))

# allowed_str = '+'.join(allowed_indices)
# print(allowed_str, len(allowed_indices))

chA, chB = list(pose.split_by_chain())[0], list(pose.split_by_chain())[1]

mpnn_design = MPNNDesign(
    design_selector=design_selector,
    omit_AAs="CX",
    temperature=0.1,
)

mpnn_design.apply(chA)

# get the mpnn sequences from the pose datacache
mpnn_seqs = {k: v for k, v in chA.scores.items() if "mpnn_seq" in k}

final_pose = pyrosetta.rosetta.core.pose.Pose()
pyrosetta.rosetta.core.pose.append_pose_to_pose(final_pose, chA, new_chain=True)
pyrosetta.rosetta.core.pose.append_pose_to_pose(final_pose, chB, new_chain=True)

# add the mpnn sequences to the final pose
for k, v in mpnn_seqs.items():
    pyrosetta.rosetta.core.pose.setPoseExtraScore(final_pose, k, v)

# get pdb_info from pose
pdb_info = pose.pdb_info()
# add pdb_info to final_pose
final_pose.pdb_info(pdb_info)
# dump the final pose
# final_pose.dump_pdb(opts.outfile)
# dump a fasta
# mpnn_design.dump_fasta(final_pose, "test_mpnn.fasta")

# also dump each mpnn sequence to a pdb
for i, p in enumerate(mpnn_design.generate_all_poses(final_pose, include_native=False)):
    p.dump_pdb(f"{opts.outdir}/{pdbname}_mpnn_{i}.pdb")
    if i == 2:
    	break
