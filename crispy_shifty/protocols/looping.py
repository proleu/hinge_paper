# Python standard library
from typing import Iterator, List, Optional, Tuple

from pyrosetta.distributed import requires_init

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose


def loop_match(pose: Pose, length: int, connections: str = "[A+B]") -> str:
    """
    :param: pose: The pose to insert the loop into.
    :param: length: The length of the loop.
    :param: connections: The connections to use.
    :return: Whether the loop was successfully inserted.
    Runs ConnectChainsMover.
    """
    import pyrosetta

    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <ConnectChainsMover name="connectchains" 
                chain_connections="{connections}" 
                loopLengthRange="{length},{length}" 
                resAdjustmentRangeSide1="0,0" 
                resAdjustmentRangeSide2="0,0" 
                RMSthreshold="1.0"/>
        </MOVERS>
        """
    )
    cc_mover = objs.get_mover("connectchains")
    try:
        cc_mover.apply(pose)
        closure_type = "loop_match"
    except RuntimeError:  # if ConnectChainsMover cannot find a closure
        closure_type = "not_closed"
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "closure_type", closure_type)
    return closure_type


def loop_extend(
    pose: Pose,
    connections: str = "[A+B]",
    extend_before: int = 3,
    extend_after: int = 3,
    min_loop_length: int = 2,
    max_loop_length: int = 5,
    rmsd_threshold: float = 0.8,
) -> str:
    """
    :param: pose: The pose to insert the loop into.
    :param: connections: The connections to use.
    :param: extend_before: The number of residues allowed to extend before the loop.
    :param: extend_after: The number of residues allowed to extend after the loop.
    :param: min_loop_length: The minimum length of the loop.
    :param: max_loop_length: The maximum length of the loop.
    :return: Whether the loop was successfully inserted.
    Runs ConnectChainsMover.
    May increase the loop length relative to the parent
    """
    import pyrosetta

    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <ConnectChainsMover name="connectchains" 
                chain_connections="{connections}" 
                loopLengthRange="{min_loop_length},{max_loop_length}" 
                resAdjustmentRangeSide1="0,{extend_before}" 
                resAdjustmentRangeSide2="0,{extend_after}" 
                RMSthreshold="{rmsd_threshold}"/>
        </MOVERS>
        """
    )
    cc_mover = objs.get_mover("connectchains")
    try:
        cc_mover.apply(pose)
        closure_type = "loop_extend"
    except RuntimeError:  # if ConnectChainsMover cannot find a closure
        closure_type = "not_closed"
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "closure_type", closure_type)
    return closure_type


def phi_psi_omega_to_abego(phi: float, psi: float, omega: float) -> str:
    """
    :param: phi: The phi angle.
    :param: psi: The psi angle.
    :param: omega: The omega angle.
    :return: The abego string.
    From Buwei
    https://wiki.ipd.uw.edu/protocols/dry_lab/rosetta/scaffold_generation_with_piecewise_blueprint_builder
    """
    if psi == None or phi == None:
        return "X"
    if omega == None:
        omega = 180

    if abs(omega) < 90:
        return "O"
    elif phi > 0:
        if -100.0 <= psi < 100:
            return "G"
        else:
            return "E"
    else:
        if -75.0 <= psi < 50:
            return "A"
        else:
            return "B"


def abego_string(phi_psi_omega: List[Tuple[float]]) -> str:
    """
    :param: phi_psi_omega: A list of tuples of phi, psi, and omega angles.
    :return: The abego string.
    From Buwei
    https://wiki.ipd.uw.edu/protocols/dry_lab/rosetta/scaffold_generation_with_piecewise_blueprint_builder
    """
    out = ""
    for x in phi_psi_omega:
        out += phi_psi_omega_to_abego(x[0], x[1], x[2])
    return out


def get_torsions(pose: Pose) -> List[Tuple[float]]:
    """
    :param: pose: The pose to get the torsions from.
    :return: A list of tuples of phi, psi, and omega angles.
    From Buwei
    https://wiki.ipd.uw.edu/protocols/dry_lab/rosetta/scaffold_generation_with_piecewise_blueprint_builder
    """
    torsions = []
    for i in range(1, pose.total_residue() + 1):
        # these error if a chain is a non-protein (DNA, for example)
        # so, put them in try/except blocks
        try:
            phi = pose.phi(i)
        except:
            phi = None
        try:
            psi = pose.psi(i)
        except:
            psi = None
        try:
            omega = pose.omega(i)
        except:
            omega = None
        if i == 1:
            phi = None
        if i == pose.total_residue():
            psi = None
            omega = None
        torsions.append((phi, psi, omega))
    return torsions


def remodel_helper(
    pose: Pose,
    loop_length: int,
    loop_dssp: Optional[str] = None,
    remodel_before_loop: int = 1,
    remodel_after_loop: int = 1,
    surround_loop_with_helix: bool = False,
    new_loop_resl: str = "V",
) -> str:
    """
    :param: pose: The pose to insert the loop into.
    :param: loop_length: The length of the fragment to insert.
    :param: loop_dssp: The dssp string of the fragment to insert.
    :param: remodel_before_loop: The number of residues to remodel before the loop.
    :param: remodel_after_loop: The number of residues to remodel after the loop.
    :return: The filename of the blueprint file to be used to remodel the pose.
    Writes a blueprint file to the current directory or TMPDIR and returns the filename.
    """

    import os
    import uuid

    import pyrosetta

    tors = get_torsions(pose)
    abego_str = abego_string(tors)
    dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(pose)
    # name blueprint a random 32 long hex string
    if "TMPDIR" in os.environ:
        tmpdir_root = os.environ["TMPDIR"]
    else:
        tmpdir_root = os.getcwd()
    filename = os.path.join(tmpdir_root, uuid.uuid4().hex + ".bp")
    # write a temporary blueprint file
    if not os.path.exists(tmpdir_root):
        os.makedirs(tmpdir_root, exist_ok=True)
    else:
        pass
    with open(filename, "w+") as f:
        end1, begin2 = (
            pose.chain_end(1),
            pose.chain_begin(2),
        )
        # end2 = pose.chain_end(2)
        end2 = (
            pose.size()
        )  # allows this to work with any number of chains, and it'll just try to loop the first two
        for i in range(1, end1 + 1):
            if i >= end1 - (remodel_before_loop - 1):
                if surround_loop_with_helix:
                    position_dssp = "H"
                else:
                    position_dssp = dssp[i - 1]
                print(
                    str(i),
                    pose.residue(i).name1(),
                    position_dssp + "X",
                    "R",
                    file=f,
                )
            else:
                print(
                    str(i),
                    pose.residue(i).name1(),
                    dssp[i - 1] + abego_str[i - 1],
                    ".",
                    file=f,
                )
        if loop_dssp is None:
            for i in range(loop_length):
                print("0", new_loop_resl, "LX", "R", file=f)
        else:
            try:
                assert len(loop_dssp) == loop_length
            except AssertionError:
                raise ValueError("loop_dssp must be the same length as loop_length")
            for i in range(loop_length):
                print(
                    "0",
                    new_loop_resl,
                    f"{loop_dssp[i]}X",
                    "R",
                    file=f,
                )
        for i in range(begin2, end2 + 1):
            if i <= begin2 + (remodel_after_loop - 1):
                if surround_loop_with_helix:
                    position_dssp = "H"
                else:
                    position_dssp = dssp[i - 1]
                print(
                    str(i),
                    pose.residue(i).name1(),
                    position_dssp + "X",
                    "R",
                    file=f,
                )
            else:
                print(
                    str(i),
                    pose.residue(i).name1(),
                    dssp[i - 1] + abego_str[i - 1],
                    ".",
                    file=f,
                )

    return filename


def loop_remodel(
    pose: Pose,
    length: int,
    attempts: int = 10,
    loop_dssp: Optional[str] = None,
    remodel_before_loop: int = 1,
    remodel_after_loop: int = 1,
    remodel_lengths_by_vector: bool = False,
    surround_loop_with_helix: bool = False,
    new_loop_resl: str = "V",
) -> str:
    """
    :param: pose: The pose to insert the loop into.
    :param: length: The length of the loop.
    :param: attempts: The number of attempts to make.
    :param: remodel_before_loop: The number of residues to remodel before the loop.
    :param: remodel_after_loop: The number of residues to remodel after the loop.
    :param: remodel_lengths_by_vector: Use the vector angles of chain ends to determine what length to remodel.
    :return: Whether the loop was successfully inserted.
    Remodel a new loop using Blueprint Builder. Loops the first two chains, leaving the others untouched as context.
    DSSP and SS agnostic in principle but in practice more or less matches.
    """
    import os

    import numpy as np
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose

    # computes the number of residues to remodel before and after the loop by finding which residue-residue vectors point towards the helix to loop to
    # probably works best for building a loop between two helices
    # still uses the default lengths to remodel if none of the vectors are good (dot>0)
    if remodel_lengths_by_vector:
        end1, begin2 = (pose.chain_end(1), pose.chain_begin(2))
        max_dot_1 = 0
        max_dot_2 = 0
        vec_12 = pose.residue(begin2).xyz("CA") - pose.residue(end1).xyz("CA")
        for i in range(3):
            vec_1 = pose.residue(end1 - i).xyz("CA") - pose.residue(end1 - i - 1).xyz(
                "CA"
            )
            dot_1 = vec_12.dot(
                vec_1.normalize()
            )  # normalization accounts for slight differences in Ca-Ca distances dependent on secondary structure
            if dot_1 > max_dot_1:
                max_dot_1 = dot_1
                remodel_before_loop = i + 1
            vec_2 = pose.residue(begin2 + i + 1).xyz("CA") - pose.residue(
                begin2 + i
            ).xyz("CA")
            dot_2 = vec_12.dot(vec_2.normalize())
            if dot_2 > max_dot_2:
                max_dot_2 = dot_2
                remodel_after_loop = i + 1

    if loop_dssp is None:
        bp_file = remodel_helper(
            pose,
            length,
            remodel_before_loop=remodel_before_loop,
            remodel_after_loop=remodel_after_loop,
            surround_loop_with_helix=surround_loop_with_helix,
            new_loop_resl=new_loop_resl,
        )
    else:
        bp_file = remodel_helper(
            pose,
            length,
            loop_dssp=loop_dssp,
            remodel_before_loop=remodel_before_loop,
            remodel_after_loop=remodel_after_loop,
            surround_loop_with_helix=surround_loop_with_helix,
            new_loop_resl=new_loop_resl,
        )

    bp_sfxn = pyrosetta.create_score_function("fldsgn_cen.wts")
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_sr_bb, 1.0)
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.hbond_lr_bb, 1.0)
    bp_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint, 1.0
    )
    bp_sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.angle_constraint, 1.0)
    bp_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.dihedral_constraint, 1.0
    )

    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <BluePrintBDR name="blueprintbdr" 
            blueprint="{bp_file}" 
            use_abego_bias="0" 
            use_sequence_bias="0" 
            rmdl_attempts="20"/>
        </MOVERS>
        """
    )
    bp_mover = objs.get_mover("blueprintbdr")
    bp_mover.scorefunction(bp_sfxn)

    closure_type = "not_closed"
    orig_num_chains = pose.num_chains()
    for _ in range(attempts):
        bp_mover.apply(pose)
        if pose.num_chains() < orig_num_chains:
            closure_type = "loop_remodel"
            break

    os.remove(bp_file)

    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "closure_type", closure_type)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        pose, "remodel_before_loop", str(remodel_before_loop)
    )
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        pose, "remodel_after_loop", str(remodel_after_loop)
    )
    return closure_type


@requires_init
def loop_complex(pose: Pose, all_chains_to_loop: list, all_loop_lengths: list):

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta

    sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    from crispy_shifty.protocols.design import (
        clear_constraints,
        gen_std_layer_design,
        gen_task_factory,
        packrotamers,
        score_ss_sc,
        score_wnm,
        struct_profile,
    )
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # all_chains_to_loop = [[int(chain) for chain in chains.split(',')] for chains in kwargs["chains_to_loop"].split(";")]
    # all_loop_lengths = [[int(length) for length in lengths.split(',')] for lengths in kwargs["loop_lengths"].split(";")]

    # For convenience, the all_chains_to_loop function argument only includes the chains to loop.
    # Add in all the other chains that aren't being looped.
    chain_loop_configuration = [([i], None) for i in range(1, all_chains_to_loop[0][0])]
    for i in range(len(all_chains_to_loop) - 1):
        chain_loop_configuration.append((all_chains_to_loop[i], all_loop_lengths[i]))
        for j in range(all_chains_to_loop[i][-1] + 1, all_chains_to_loop[i + 1][0]):
            chain_loop_configuration.append(([j], None))
    chain_loop_configuration.append((all_chains_to_loop[-1], all_loop_lengths[-1]))
    for j in range(all_chains_to_loop[-1][-1] + 1, pose.num_chains() + 1):
        chain_loop_configuration.append(([j], None))

    # careful- all_chains is still 1-indexed, so the chains_to_loop kwarg is 1-indexed also
    all_chains = pose.split_by_chain()

    looped_poses = []
    closure_type = "not_closed"
    for chains_to_loop, loop_lengths in chain_loop_configuration:
        looped_pose = all_chains[chains_to_loop[0]]

        if len(chains_to_loop) >= 2:
            new_loop_strs = []
            for unlooped_chain, loop_length in zip(chains_to_loop[1:], loop_lengths):
                loop_start = int(looped_pose.size()) + 1
                pyrosetta.rosetta.core.pose.append_pose_to_pose(
                    looped_pose, all_chains[unlooped_chain], True
                )
                # Rebuild PDBInfo for ConnectChainsMover
                pdb_info = pyrosetta.rosetta.core.pose.PDBInfo(looped_pose)
                looped_pose.pdb_info(pdb_info)

                print_timestamp(
                    "Attempting closure by loop match...", start_time, end=""
                )
                closure_type = loop_match(looped_pose, loop_length)
                # closure by loop matching was successful, move on to the next set to close or continue to scoring
                # should I use an additional check like 'pose_to_loop.num_chains() == 1' to determine if the pose is closed?
                if closure_type != "not_closed":
                    print("success.")
                else:
                    print("failed.")

                    print_timestamp(
                        "Attempting closure by loop remodel...", start_time, end=""
                    )
                    closure_type = loop_remodel(
                        pose=looped_pose,
                        length=loop_length,
                        attempts=10,
                        remodel_before_loop=1,
                        remodel_after_loop=1,
                        remodel_lengths_by_vector=True,
                    )
                    if closure_type != "not_closed":
                        print("success.")
                    else:
                        print("failed. Exiting.")
                        # couldn't close this pair; stop trying with the whole set
                        break

                # is this naive? Phil did something more complicated with residue selectors, looking at the valines.
                # Wondering if I'm missing some edge cases for which this approach doesn't work.
                new_loop_strs.append(
                    ",".join(
                        str(resi)
                        for resi in range(loop_start, loop_start + loop_length)
                    )
                )

            if closure_type == "not_closed":
                # couldn't close this set; stop trying with the all the sets
                break

            new_loop_str = ",".join(new_loop_strs)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                looped_pose, "new_loop_resis", new_loop_str
            )

        looped_poses.append(looped_pose)

    # if we couldn't close one of the sets in this complex, continue to the next pose and skip scoring and yielding the pose (so nothing is written to disk)
    if closure_type == "not_closed":
        return

    # The code will only reach here if all loops are closed.
    # Loop closure is fast but has a somewhat high failure rate, so more efficient to first see if all loops can be closed,
    # and only design and score if so.
    # First combine all the looped poses into one pose, then design the new loop residues. This ensures the new loop
    # residues are designed in the context of the rest of the pose.
    # looped_poses contains single chains of the unlooped chains, and the looped chains.

    # combine all the looped poses into one pose
    print_timestamp("Collecting poses...", start_time, end="")
    combined_looped_pose = pyrosetta.rosetta.core.pose.Pose()
    new_loop_strs = []
    loop_scores = {}
    for i, looped_pose in enumerate(looped_poses):
        chain_id = str(i + 1)
        for key, value in looped_pose.scores.items():
            loop_scores[key + "_" + chain_id] = value
        pose_end = combined_looped_pose.size()
        if "new_loop_resis" in looped_pose.scores:
            new_loop_resis = [
                int(i) + pose_end
                for i in looped_pose.scores["new_loop_resis"].split(",")
            ]
            new_loop_strs.append(",".join([str(i) for i in new_loop_resis]))
        pyrosetta.rosetta.core.pose.append_pose_to_pose(
            combined_looped_pose, looped_pose, True
        )
    new_loop_str = ",".join(new_loop_strs)
    # Rebuild PDBInfo
    pdb_info = pyrosetta.rosetta.core.pose.PDBInfo(combined_looped_pose)
    combined_looped_pose.pdb_info(pdb_info)
    # Add scores from looping
    for key, value in loop_scores.items():
        pyrosetta.rosetta.core.pose.setPoseExtraScore(combined_looped_pose, key, value)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        combined_looped_pose, "new_loop_resis", new_loop_str
    )
    print("complete.")
    print(new_loop_str)

    print_timestamp("Designing loops...", start_time, end="")
    layer_design = gen_std_layer_design()
    design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
    )

    new_loop_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
        new_loop_str
    )
    design_sel = (
        pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
            new_loop_sel, 6, True
        )
    )
    task_factory = gen_task_factory(
        design_sel=design_sel,
        pack_nbhd=True,
        extra_rotamers_level=2,
        limit_arochi=True,
        prune_buns=True,
        upweight_ppi=False,
        restrict_pro_gly=False,
        precompute_ig=True,
        ifcl=True,
        layer_design=layer_design,
    )
    struct_profile(
        combined_looped_pose, design_sel
    )  # Phil's code used eliminate_background=False...
    packrotamers(combined_looped_pose, task_factory, design_sfxn)
    clear_constraints(combined_looped_pose)
    print("complete.")

    # Score the packed loops
    print_timestamp("Scoring...", start_time, end="")
    for i, looped_pose in enumerate(combined_looped_pose.split_by_chain()):
        chain_id = str(i + 1)
        total_length = len(looped_pose.residues)
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            combined_looped_pose, "total_length_" + chain_id, total_length
        )
        dssp = pyrosetta.rosetta.protocols.simple_filters.dssp(looped_pose)
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            combined_looped_pose, "dssp_" + chain_id, dssp
        )
        tors = get_torsions(looped_pose)
        abego_str = abego_string(tors)
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            combined_looped_pose, "abego_str_" + chain_id, abego_str
        )

        # should be fast since the database is already loaded from CCM/SPM
        wnm = score_wnm(looped_pose)
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            combined_looped_pose, "wnm_" + chain_id, wnm
        )
        ss_sc = score_ss_sc(looped_pose, False, True, "loop_sc")
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            combined_looped_pose, "ss_sc_" + chain_id, ss_sc
        )
    print("complete.")

    return combined_looped_pose


@requires_init
def loop_all_hinges(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:

    import sys

    import pyrosetta
    import pyrosetta.distributed.io as io

    sys.path.insert(0, "/mnt/home/broerman/projects/crispy_shifty")
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    all_chains_to_loop = [
        [int(chain) for chain in chains.split(",")]
        for chains in kwargs["chains_to_loop"].split("/")
    ]

    for pose in poses:
        scores = dict(pose.scores)
        pyrosetta.rosetta.core.pose.clearPoseExtraScores(pose)

        parent_length = int(float(scores["parent_length"]))
        loop_length = int(parent_length - pose.chain_end(2))

        looped_pose = loop_complex(
            pose,
            all_chains_to_loop,
            [[loop_length] for _ in range(len(all_chains_to_loop))],
        )

        if looped_pose is not None:
            # Add old scores back into the pose
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, key, value)

            ppose = io.to_packed(looped_pose)
            yield ppose


@requires_init
def loop_bound_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be looped.
    :param: kwargs: keyword arguments to be passed to looping protocol.
    :return: an iterator of PackedPose objects.
    Assumes that pyrosetta.init() has been called with `-corrections:beta_nov16` .
    `-indexed_structure_store:fragment_store \
    /net/databases/VALL_clustered/connect_chains/ss_grouped_vall_helix_shortLoop.h5`
    """

    import sys
    from copy import deepcopy
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        clear_constraints,
        clear_terms_from_scores,
        gen_std_layer_design,
        gen_task_factory,
        pack_rotamers,
        score_ss_sc,
        score_wnm,
        struct_profile,
    )
    from crispy_shifty.protocols.states import clash_check
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    looped_poses = []
    sw = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    for pose in poses:
        scores = dict(pose.scores)
        bb_clash_pre = clash_check(pose)
        # get parent length from the scores
        parent_length = int(float(scores["trimmed_length"]))
        looped_poses = []
        sw.chain_order("123")
        sw.apply(pose)
        min_loop_length = int(parent_length - pose.chain_end(2))
        max_loop_length = 5
        loop_start = pose.chain_end(1) + 1
        pre_looped_length = pose.chain_end(2)
        print_timestamp("Generating loop extension...", start_time, end="")
        closure_type = loop_extend(
            pose=pose,
            connections="[A+B],C",
            extend_before=3,
            extend_after=3,
            min_loop_length=min_loop_length,
            max_loop_length=max_loop_length,
            rmsd_threshold=0.5,
        )
        if closure_type == "not_closed":
            continue  # move on to next pose, we don't care about the ones that aren't closed
        else:
            sw.chain_order("12")
            sw.apply(pose)
            # get new loop resis
            new_loop_length = pose.chain_end(1) - pre_looped_length
            new_loop_str = ",".join(
                [str(i) for i in range(loop_start, loop_start + new_loop_length)]
            )
            bb_clash_post = clash_check(pose)
            scores["bb_clash_delta"] = bb_clash_post - bb_clash_pre
            scores["new_loop_str"] = new_loop_str
            scores["looped_length"] = pose.chain_end(1)
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            looped_poses.append(pose)

    layer_design = gen_std_layer_design()
    design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
    )

    for looped_pose in looped_poses:
        scores = dict(looped_pose.scores)
        new_loop_str = scores["new_loop_str"]
        # don't design any fixed residues
        fixed_sel = (
            pyrosetta.rosetta.core.select.residue_selector.FalseResidueSelector()
        )
        if "fixed_resis" in scores:
            fixed_resi_str = scores["fixed_resis"]
            # handle an empty string
            if fixed_resi_str:
                # adjust the fixed residue indices to account for the new loop
                new_loop_resis = [int(i) for i in new_loop_str.split(",")]
                new_loop_len = len(new_loop_resis)
                fixed_resis = [int(resi) for resi in fixed_resi_str.split(",")]
                for fixed_i, resi in enumerate(fixed_resis):
                    if resi >= new_loop_resis[0]:
                        fixed_resis[fixed_i] += new_loop_len
                fixed_resi_str = ",".join(map(str, fixed_resis))
                scores["fixed_resis"] = fixed_resi_str
                fixed_sel = (
                    pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
                        fixed_resi_str
                    )
                )
        print_timestamp("Building loop...", start_time)
        new_loop_sel = (
            pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
                new_loop_str
            )
        )
        design_sel = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(
            pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
                new_loop_sel, 8, True
            ),
            pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(
                fixed_sel
            ),
        )
        task_factory = gen_task_factory(
            design_sel=design_sel,
            pack_nbhd=True,
            extra_rotamers_level=2,
            limit_arochi=True,
            prune_buns=False,
            upweight_ppi=False,
            restrict_pro_gly=False,
            precompute_ig=False,
            ifcl=True,
            layer_design=layer_design,
        )
        print_timestamp("Designing loop...", start_time)
        struct_profile(
            looped_pose,
            design_sel,
        )
        # pack the loop twice
        for _ in range(0, 2):
            pack_rotamers(
                looped_pose,
                task_factory,
                design_sfxn,
            )
        clear_constraints(looped_pose)
        print_timestamp("Scoring loop...", start_time)
        pyrosetta.rosetta.core.pose.clearPoseExtraScores(looped_pose)
        total_length = looped_pose.total_residue()
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            looped_pose, "total_length", total_length
        )
        dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(looped_pose)
        pyrosetta.rosetta.core.pose.setPoseExtraScore(
            looped_pose, "dssp", dssp.get_dssp_secstruct()
        )
        score_ss_sc(looped_pose, False, True, "loop_sc")
        scores.update(looped_pose.scores)
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(looped_pose, key, value)
        clear_terms_from_scores(looped_pose)
        ppose = io.to_packed(looped_pose)
        yield ppose
