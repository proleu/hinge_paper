# Python standard library
from typing import Dict, Iterator, List, Optional, Union

from pyrosetta.distributed import requires_init

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose

# Custom library imports


@requires_init
def mpnn_dhr(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNDesign, or this function.
    :return: an iterator of PackedPose objects.
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import interface_between_selectors
    from crispy_shifty.protocols.mpnn import MPNNDesign
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=False, pack_result=False
        )

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        print_timestamp("Setting up design selector", start_time)
        # MAKE A SELECTOR THAT SELECTS THE RESIDUES TO BE DESIGNED
        DESIGN_SELECTOR = ChainSelector(1)
        print_timestamp("Designing with MPNN", start_time)
        # construct the MPNNDesign object
        mpnn_design = MPNNDesign(
            design_selector=DESIGN_SELECTOR,
            # MPNN understands layers, unsats, helix caps etc so just ban CYS
            omit_AAs="CX",
            **kwargs,
        )
        # design the pose
        mpnn_design.apply(pose)
        print_timestamp("MPNN design complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        scores["path_in"] = pdb_path
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # generate the original pose, with the sequences written to the datacache
        ppose = io.to_packed(pose)
        yield ppose


@requires_init
def fold_dhr(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to fold with the superfold script.
    :param: kwargs: keyword arguments to be passed to the superfold script.
    :return: an iterator of PackedPose objects.
    """

    import sys
    from operator import gt, lt
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.folding import (
        SuperfoldRunner,
        generate_decoys_from_pose,
    )
    from crispy_shifty.utils.io import cmd, print_timestamp

    start_time = time()
    # hacky split pdb_path into pdb_path and fasta_path
    pdb_path = kwargs.pop("pdb_path")
    pdb_path, fasta_path = tuple(pdb_path.split("____"))

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        # skip the kwargs check
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        print_timestamp("Setting up for AF2", start_time)
        runner = SuperfoldRunner(pose=pose, fasta_path=fasta_path, **kwargs)
        runner.setup_runner(file=fasta_path)
        # reference_pdb is the tmp.pdb dumped by SuperfoldRunner for computing RMSD
        reference_pdb = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        flag_update = {
            "--reference_pdb": reference_pdb,
        }
        runner.update_flags(flag_update)
        runner.update_command()
        print_timestamp("Running AF2", start_time)
        # add scores for the different sequences to the pose datacache
        runner.apply(pose)
        print_timestamp("AF2 complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # set up the filters to get only the best sequences
        filter_dict = {
            "mean_plddt": (gt, 90.0),
            "rmsd_to_reference": (lt, 1.5),
        }
        # rank the sequences by mean plddt (only matters if multiple models were used)
        rank_on = "mean_plddt"
        # look for sequences that had this prefix
        prefix = "mpnn_seq"
        # generate poses with sequences that pass the filters
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            label_first=True,  # the first sequence is the original rosetta sequence
            prefix=prefix,
            rank_on=rank_on,
        ):
            ppose = io.to_packed(decoy)
            yield ppose


@requires_init
def mpnn_dhr_with_repeat(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNMultistateDesign, or this
    function.
    :return: an iterator of PackedPose objects.
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        ResidueIndexSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import interface_between_selectors
    from crispy_shifty.protocols.mpnn import MPNNMultistateDesign
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=False, pack_result=False
        )

    internal_repeats_dict = {
        "DHR14": ((38, 77), (78, 117)),
        "DHR15": ((44, 89), (90, 135)),
        "DHR27": ((52, 105), (106, 159)),
        "DHR59": ((47, 94), (95, 142)),
        "DHR68": ((59, 119), (120, 180)),
        "DHR70": ((49, 98), (99, 148)),
        "DHR81": ((56, 112), (113, 169)),
        "DHR82": ((59, 116), (117, 174)),
    }

    # get the name of the DHR in a hacky way, reccomend using pathlib for better parsing
    dhr = pdb_path.split("/")[-1].split(".")[0]
    # get the internal repeats
    internal_repeats = internal_repeats_dict[dhr]
    # each tuple in internal_repeats is a tuple of (start, end)
    # we want to make a selector for each of these as a comma separated string of all
    # residues in the range start-end
    repeat_string_1 = ",".join(
        [str(i) for i in range(internal_repeats[0][0], internal_repeats[0][1] + 1)]
    )
    repeat_string_2 = ",".join(
        [str(i) for i in range(internal_repeats[1][0], internal_repeats[1][1] + 1)]
    )

    # poses should contian only one pose in this case we will keep the loop below anyway
    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        scores["original_sequence"] = pose.sequence()
        print_timestamp("Setting up design selector", start_time)
        # MAKE A SELECTOR THAT SELECTS THE RESIDUES TO BE DESIGNED
        design_selector = ChainSelector(1)
        # MAKE SELECTORS THAT SELECT THE RESIDUE GROUPS TO BE LINKED
        residue_selectors = [
            [
                ResidueIndexSelector(repeat_string_1),
                ResidueIndexSelector(repeat_string_2),
            ]
        ]
        print_timestamp("Designing with MPNN", start_time)
        # construct the MPNNMultistateDesign object
        mpnn_design = MPNNMultistateDesign(
            design_selector=design_selector,
            residue_selectors=residue_selectors,
            omit_AAs="CX",
            **kwargs,
        )
        # design the pose
        mpnn_design.apply(pose)
        print_timestamp("MPNN design complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        scores["path_in"] = pdb_path
        scores["parent"] = dhr
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # generate all output poses
        for decoy in mpnn_design.generate_all_poses(pose, include_native=True):
            ppose = io.to_packed(decoy)
            yield ppose


@requires_init
def batch_fold_dhr(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to fold with the superfold script.
    :param: kwargs: keyword arguments to be passed to the superfold script.
    :return: an iterator of PackedPose objects.
    """

    import sys
    from operator import gt, lt
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import score_SAP
    from crispy_shifty.protocols.folding import (
        SuperfoldMultiPDB,
        generate_decoys_from_pose,
    )
    from crispy_shifty.utils.io import cmd, print_timestamp

    start_time = time()
    # hacky split pdb_path into pdb_path and fasta_path
    pdb_path = kwargs.pop("pdb_path")
    # there are multiple paths in the pdb_path, we need to split them and rejoin them
    pdb_paths = pdb_path.split("____")
    pdb_path = " ".join(pdb_paths)

    # this function is special, we don't want a packed_pose_in ever, we maintain it as
    # a kwarg for backward compatibility with PyRosettaCluster
    if packed_pose_in is not None:
        raise ValueError("This function is not intended to have a packed_pose_in")
    else:
        pass

    print_timestamp("Setting up for AF2", start_time)
    runner = SuperfoldMultiPDB(input_file=pdb_path, load_decoys=True, **kwargs)
    runner.setup_runner()
    print_timestamp("Running AF2", start_time)
    runner.apply()
    print_timestamp("AF2 complete, updating pose datacache", start_time)
    # get the updated poses from the runner
    tag_pose_dict = runner.get_tag_pose_dict()
    # rank the different model results by plddt
    rank_on = "mean_plddt"
    print_timestamp("Generating decoys", start_time)
    for tag, pose in tag_pose_dict.items():
        for decoy in generate_decoys_from_pose(
            pose,
            generate_prediction_decoys=True,
            label_first=False,
            prefix=tag,
            rank_on=rank_on,
        ):
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                decoy, "final_seq", decoy.sequence()
            )
            sap_score = score_SAP(decoy)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(decoy, "sap_score", sap_score)
            final_ppose = io.to_packed(decoy)
            yield final_ppose
