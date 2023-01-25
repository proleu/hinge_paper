# Python standard library
from typing import Dict, Iterator, List, Optional, Union

# 3rd party library imports
from pyrosetta.distributed import requires_init

# Rosetta library imports
import pyrosetta
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector


def score_rmsd(
    pose: Pose,
    refpose: Pose,
    sel: ResidueSelector = None,
    refsel: ResidueSelector = None,
    rmsd_type: pyrosetta.rosetta.core.scoring.rmsd_atoms = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca,
    name: str = "rmsd",
):
    # written by Adam Broerman
    rmsd_metric = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
    rmsd_metric.set_comparison_pose(refpose)
    if sel == None:
        sel = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    rmsd_metric.set_residue_selector(sel)
    if refsel == None:
        refsel = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    rmsd_metric.set_residue_selector_reference(refsel)
    rmsd_metric.set_rmsd_type(rmsd_type)  # Default is rmsd_all_heavy
    rmsd_metric.set_run_superimpose(True)
    rmsd = rmsd_metric.calculate(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, rmsd)
    return rmsd


def find_top_alignment_index(a,b):
    """
    Finds the index of top alignment for b in a. 
    Top alignment meaning the alignment where the most entries in b match 
    their currently aligned counterparts in a 
    """
    top_score = 0
    top_idx = -1
    L = len(b)
    
    for i in range(len(a) - L + 1):
        
        chunk = a[i:i+L]
        
        cur_score = 0
        for b_char, chunk_char in zip(b, chunk):
            if b_char == chunk_char:
                cur_score += 1
        
        if cur_score > top_score:
            top_idx = i
            top_score = cur_score

    return top_idx


def model_hinge_alt_state(
    pose: Pose,
    alt_state: Pose, # hinge alt state should be chain 1
    end_N_side: int, # for our hinges, should be the last helical residue before the new loop
    start_C_side: int, # for our hinges, should be the first helical residue after the new loop
):
    # written by Adam Broerman
    from copy import deepcopy
    from crispy_shifty.protocols.states import range_CA_align, fuse_two_poses

    pose_chains = list(pose.split_by_chain())
    fusion_pose = pose_chains[0]

    # find top sequence alignment of hinge
    top_idx = find_top_alignment_index(fusion_pose.sequence(), alt_state.chain_sequence(1))

    # align alt_state to fusion_pose from resi 1 to end_N_side
    range_CA_align(alt_state, fusion_pose, 1, end_N_side, 1+top_idx, end_N_side+top_idx)

    # extract C-terminal side of pose into a new C-pose
    # hacky leave an extra residue so the actual first residue is not the Nterm variant
    extract_C_side = pyrosetta.rosetta.protocols.grafting.simple_movers.DeleteRegionMover(1, start_C_side-2)
    extract_C_side.set_rechain(True)
    C_pose = deepcopy(fusion_pose)
    extract_C_side.apply(C_pose)

    # align C-pose to alt_state from start of next helix to end of alt_state
    range_CA_align(C_pose, alt_state, 2, alt_state.chain_end(1)-start_C_side+2, start_C_side, alt_state.chain_end(1))

    # rebuild pose in alt state with the loop from alt_state
    alt_pose = fuse_two_poses(fusion_pose, alt_state, end_N_side+top_idx, end_N_side+1, 1, start_C_side-1)
    # start_b from non-Nterm variant actual first residue
    alt_pose = fuse_two_poses(alt_pose, C_pose, start_C_side+top_idx-1, 2)

    # add any additional chains from the original pose
    for additional_chain in pose_chains[1:]:
        pyrosetta.rosetta.core.pose.append_pose_to_pose(alt_pose, additional_chain, True)

    # add any additional chains from alt_state to pose
    for additional_chain in list(alt_state.split_by_chain())[1:]:
        pyrosetta.rosetta.core.pose.append_pose_to_pose(alt_pose, additional_chain, True)
    
    return alt_pose


def add_interaction_partner(
    pose: Pose,
    int_state: Pose,
    int_chain: int,
):
    # written by Adam Broerman
    from crispy_shifty.protocols.states import range_CA_align

    alt_pose = pose.clone()

    # find top sequence alignment of the chain of the interaction that is in the pose
    top_idx = find_top_alignment_index(alt_pose.sequence(), int_state.chain_sequence(int_chain))

    # align int_state to pose
    range_CA_align(int_state, alt_pose, int_state.chain_begin(int_chain), int_state.chain_end(int_chain), 1+top_idx, int_state.chain_end(int_chain)-int_state.chain_begin(int_chain)+1+top_idx)

    # add additional chains of the interaction from int_state to pose
    for i, additional_chain in enumerate(list(int_state.split_by_chain()), start=1):
        if i != int_chain:
            pyrosetta.rosetta.core.pose.append_pose_to_pose(alt_pose, additional_chain, True)
    
    return alt_pose


def rebuild_component(
    pose: Pose,
    original_pose: Pose,
    align_depth: int, # the last residue of original_pose remaining in the fusion counting from the terminus fused to
    fusion_terminus: str # the terminus fused to (opposite the free terminus we're restoring)
):
    # written by Adam Broerman and Abbas Idris
    # The idea is that fusion_pose may have been changed by some function, and we want to rebuild the component while
    # preserving those changes. So we keep as much of fusion_pose as possible.
    
    from crispy_shifty.protocols.states import range_CA_align, fuse_two_poses

    pose_chains = list(pose.split_by_chain())
    fusion_pose = pose_chains[0]

    if fusion_terminus == "N":
        # find top sequence alignment of component
        top_idx = find_top_alignment_index(fusion_pose.sequence(), original_pose.chain_sequence(1)[:align_depth])
        range_CA_align(original_pose, fusion_pose, 1, align_depth, 1+top_idx, align_depth+top_idx)
        fusion_len = fusion_pose.chain_end(1)
        fusion_pose = fuse_two_poses(fusion_pose, original_pose, fusion_len-1, fusion_len-top_idx, end_b=original_pose.chain_end(1))
    elif fusion_terminus == "C":
        top_idx = find_top_alignment_index(fusion_pose.sequence(), original_pose.chain_sequence(1)[-align_depth:])
        original_len = original_pose.chain_end(1)
        range_CA_align(original_pose, fusion_pose, original_len-align_depth+1, original_len, 1+top_idx, top_idx+align_depth)
        # top_idx + align_depth is the last residue of the original_pose that is in the fusion_pose
        fusion_pose = fuse_two_poses(original_pose, fusion_pose, original_len-(top_idx+align_depth)+1, 2)
    else:
        raise ValueError("fusion_terminus must be either 'N' or 'C'")

    # add any additional chains from the original pose
    for additional_chain in pose_chains[1:]:
        pyrosetta.rosetta.core.pose.append_pose_to_pose(fusion_pose, additional_chain, True)

    # add any additional chains from alt_state to pose
    for additional_chain in list(original_pose.split_by_chain())[1:]:
        pyrosetta.rosetta.core.pose.append_pose_to_pose(fusion_pose, additional_chain, True)
    
    return fusion_pose


def symmetrize_by_alignment(
    pose: Pose,
    symmetry_pose: Pose,
    align_depth: int, # the last residue of symmetry_pose remaining in the fusion counting from the terminus fused to
    fusion_terminus: str # the terminus the oligomer is fused to
):
    # written by Adam Broerman and Abbas Idris
    from copy import deepcopy
    from crispy_shifty.protocols.states import range_CA_align

    symmetry_chains = list(symmetry_pose.split_by_chain())

    full_sym_pose = pyrosetta.rosetta.core.pose.Pose()
    for ref_chain in symmetry_chains:
        sym_pose = deepcopy(pose)
        if fusion_terminus == "N":
            range_CA_align(sym_pose, ref_chain, 1, align_depth, 1, align_depth)
        elif fusion_terminus == "C":
            ref_len = ref_chain.chain_end(1)
            sym_len = sym_pose.chain_end(1)
            range_CA_align(sym_pose, ref_chain, sym_len-align_depth, sym_len, ref_len-align_depth, ref_len)
        else:
            raise ValueError("fusion_terminus must be either 'N' or 'C'")

        for chain in sym_pose.split_by_chain():
            pyrosetta.rosetta.core.pose.append_pose_to_pose(full_sym_pose, chain, True)
    
    return full_sym_pose


@requires_init
def rebuild_and_symmetrize_fusions(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object.
    :param: kwargs: keyword arguments to be passed.
    :return: an iterator of PackedPose objects.
    """

    from pathlib import Path
    import sys
    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        cluster_scores = kwargs.pop("df_scores", True)
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=cluster_scores, pack_result=False
        )

    component_path = kwargs["component_path"]
    sym_path = kwargs["sym_path"]
    align_depth_rebuild = int(kwargs["align_depth_rebuild"])
    align_depth_sym = int(kwargs["align_depth_sym"])
    fusion_terminus = kwargs["fusion_terminus"]

    component_pose = pyrosetta.pose_from_file(component_path)
    sym_pose = pyrosetta.pose_from_file(sym_path)

    for pose in poses:
        scores = dict(pose.scores)
        scores["align_depth_rebuild"] = align_depth_rebuild
        scores["align_depth_sym"] = align_depth_sym
        scores["fusion_terminus"] = fusion_terminus

        # rebuild the component after inpainting
        pose = rebuild_component(pose, component_pose, align_depth_rebuild, fusion_terminus)
        # symmetrize the pose
        pose = symmetrize_by_alignment(pose, sym_pose, align_depth_sym, fusion_terminus)

        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        ppose = io.to_packed(pose)
        yield ppose



def find_helical_vector(xyz_ar, x0=[1,1,2,0,0,0]):
    import numpy as np
    from scipy.optimize import minimize
    #calculate perpendicular distance from vector to atom
    def get_d(a, b, xi):
        r=a+b
        s=2*a+b
        rp=xi-r
        rs=s-r
        proj_of_rp_on_rs = (np.dot(rp, rs)/np.dot(rs,rs))*rs
        di=np.sqrt(sum([x**2 for x in rp-proj_of_rp_on_rs]))
        return di
    #rmsd from average distance
    def get_val(x0,xyz_ar):
        # a,b=x0
        ds=np.asarray([get_d(x0[0:3], x0[3:], xi) for xi in xyz_ar])
        mean_d=np.mean(ds)
        val=np.sqrt(sum([(d-mean_d)**2 for d in ds]))
        return val
    #minimize
    res=minimize(get_val, x0, args=(xyz_ar))
    a=res.x[0:3]
    b=res.x[3:]
    
    #recenter on helix
    def new_ori(x0,a,b,xyz_ar):
        new_ori=a*x0+b
        def avg_dist(ori, xyz_ar):
            return np.mean([np.sqrt(sum([(xi[i]-ori[i])**2 for i in range(0,3)])) for xi in xyz_ar])
        return avg_dist(new_ori, xyz_ar)
    res2=minimize(new_ori,1,args=(a,b,xyz_ar))
    
    vec_a=a
    vec_b=b+a*res2.x[0]
    
    return (vec_a,vec_b)

def parallelness(vec_a, vec_b):
    import numpy as np
    mag_a=np.sqrt(sum([x**2 for x in vec_a]))
    mag_b=np.sqrt(sum([x**2 for x in vec_b]))
    return np.dot(vec_a/mag_a, vec_b/mag_b)
