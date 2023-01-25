# Python standard library
from typing import Iterator, Optional

from pyrosetta.distributed import requires_init

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports

target_dict = {
    "APOE": "QARLGADMEDVCGRLVQ",
    "GIP": "YAEGTFISDYSIAMDKIHQQDFVNWLLAQKGKKNDWLHNITQ",
    "Glicentin": "RSLQDTEELSRSFSASQADPLSD",  # commercial starts w PDQMNED
    "GLP1": "HAGTFTSDVSSYLEGQAAKEFIAWLVKGRG",  # core version missing first and last 2 residues # 7-37
    "GLP2": "HADGSFSDEMNTILDNLAARDFINWLIQTKITD",
    "Glucagon": "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT",
    "NPY_9-35": "GEDAPAEDMARYYSALRHYINLITRQR",  # commercial starts w SKPDNP
    # "Oxyntomodulin": "HSQGTFTSAYSKYLDSRRAQDFVQWLMNTKRNRNNIA", # nearly identical to Glucagon except for D->A mutation
    "PTH": "SVSEIQLMHNLGKHLNSMERVEWLRKKLQDVHNF",  # commercial ends with VALG
    "Secretin": "HSDGTFTSELSRLREGARLQRLLQGLV",
}


@requires_init
def mpnn_binder(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNDesign, or this function.
    :return: an iterator of PackedPose objects.
    """

    import sys
    from itertools import product
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        NeighborhoodResidueSelector,
        TrueResidueSelector,
    )

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
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    if "mpnn_temperature" in kwargs:
        if kwargs["mpnn_temperature"] == "scan":
            mpnn_temperatures = [0.1, 0.2, 0.5]
        else:
            mpnn_temperature = float(kwargs["mpnn_temperature"])
            assert (
                0.0 <= mpnn_temperature <= 1.0
            ), "mpnn_temperature must be between 0 and 1"
            mpnn_temperatures = [mpnn_temperature]
    else:
        mpnn_temperatures = [0.1]

    if "omit_AAs" in kwargs:
        omit_AAs = kwargs.pop("omit_AAs")
    else:
        omit_AAs = "X"
    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    interface_selector = AndResidueSelector(interface_between_selectors(chA, chB), chA)
    neighborhood_selector = AndResidueSelector(
        NeighborhoodResidueSelector(
            interface_selector, distance=8.0, include_focus_in_subset=True
        ),
        chA,
    )
    full_selector = chA
    selector_options = {
        "full": full_selector,
        "interface": interface_selector,
        "neighborhood": neighborhood_selector,
    }
    # make the inverse dict of selector options
    selector_inverse_options = {value: key for key, value in selector_options.items()}
    if "mpnn_design_area" in kwargs:
        if kwargs["mpnn_design_area"] == "scan":
            mpnn_design_areas = [
                selector_options[key] for key in ["full", "interface", "neighborhood"]
            ]
        else:
            try:
                mpnn_design_areas = [selector_options[kwargs["mpnn_design_area"]]]
            except:
                raise ValueError(
                    "mpnn_design_area must be one of the following: full, interface, neighborhood"
                )
    else:
        mpnn_design_areas = [selector_options["interface"]]

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        # iterate over the mpnn parameter combinations
        mpnn_conditions = list(product(mpnn_temperatures, mpnn_design_areas))
        num_conditions = len(list(mpnn_conditions))
        print_timestamp(f"Beginning {num_conditions} MPNNDesign runs", start_time)
        for i, (mpnn_temperature, mpnn_design_area) in enumerate(list(mpnn_conditions)):
            pose = original_pose.clone()
            print_timestamp(
                f"Beginning MPNNDesign run {i+1}/{num_conditions}", start_time
            )
            print_timestamp("Designing interface with MPNN", start_time)
            # construct the MPNNDesign object
            mpnn_design = MPNNDesign(
                design_selector=mpnn_design_area,
                omit_AAs=omit_AAs,
                temperature=mpnn_temperature,
                **kwargs,
            )
            # design the pose
            mpnn_design.apply(pose)
            print_timestamp("MPNN design complete, updating pose datacache", start_time)
            # update the scores dict
            scores.update(pose.scores)
            scores.update(
                {
                    "mpnn_temperature": mpnn_temperature,
                    "mpnn_design_area": selector_inverse_options[mpnn_design_area],
                }
            )
            # update the pose with the updated scores dict
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            # generate the original pose, with the sequences written to the datacache
            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def fold_binder(
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
    from pyrosetta.rosetta.core.pose import Pose

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.folding import (
        SuperfoldRunner,
        generate_decoys_from_pose,
    )
    from crispy_shifty.protocols.mpnn import dict_to_fasta, fasta_to_dict
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
        runner = SuperfoldRunner(
            pose=pose, fasta_path=fasta_path, load_decoys=True, **kwargs
        )
        runner.setup_runner(file=fasta_path)
        # initial_guess, reference_pdb both are the tmp.pdb
        initial_guess = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        reference_pdb = initial_guess
        flag_update = {
            "--initial_guess": initial_guess,
            "--reference_pdb": reference_pdb,
        }
        runner.update_flags(flag_update)
        runner.update_command()
        print_timestamp("Running AF2", start_time)
        runner.apply(pose)
        print_timestamp("AF2 complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # setup prefix, rank_on, filter_dict (in this case we can't get from kwargs)
        filter_dict = {
            "mean_pae_interaction": (lt, 10),
            "mean_plddt": (gt, 92.0),
            "pTMscore": (gt, 0.8),
            "rmsd_to_reference": (lt, 1.75),
        }
        rank_on = "mean_plddt"
        prefix = "mpnn_seq"
        print_timestamp("Generating decoys", start_time)
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            generate_prediction_decoys=True,
            label_first=True,
            prefix=prefix,
            rank_on=rank_on,
        ):
            packed_decoy = io.to_packed(decoy)
            yield packed_decoy


@requires_init
def fold_unbound(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to fold with the Superfold script.
    :param: kwargs: keyword arguments to be passed to the Superfold script.
    :return: an iterator of PackedPose objects.
    """

    import os
    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import Pose

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.mpnn import dict_to_fasta, fasta_to_dict
    from crispy_shifty.utils.io import cmd, print_timestamp

    start_time = time()
    # get the pdb_path from the kwargs
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
    runner.setup_runner(chains_to_keep=[1])
    print_timestamp("Running AF2", start_time)
    runner.apply()
    print_timestamp("AF2 complete, updating pose datacache", start_time)
    # get the updated poses from the runner
    tag_pose_dict = runner.get_tag_pose_dict()
    # filter the decoys
    filter_dict = {
        "mean_plddt": (gt, 90.0),
        "rmsd_to_input": (lt, 2.0),
    }
    rank_on = "mean_plddt"
    print_timestamp("Generating decoys", start_time)
    for tag, pose in tag_pose_dict.items():
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            generate_prediction_decoys=False,
            label_first=False,
            prefix=tag,
            rank_on=rank_on,
        ):
            scores = dict(decoy.scores)
            bound_pose = None
            for original_path in pdb_paths:
                if tag in original_path:
                    bound_pose = next(
                        path_to_pose_or_ppose(
                            path=original_path, cluster_scores=True, pack_result=False
                        )
                    )
                    final_pose = Pose()
                    break
                else:
                    continue
            if bound_pose is None:
                raise RuntimeError
            else:
                pass
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(bound_pose, key, value)
            final_ppose = io.to_packed(final_pose)
            yield final_ppose


@requires_init
def thread_target(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to thread a target peptide onto.
    :param: kwargs: keyword arguments to be passed to the threading protocol.
    :return: an iterator of PackedPose objects.
    """
    import sys
    from itertools import combinations
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        ResidueNameSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import interface_between_selectors
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
    # see if there are contact filters in kwargs, would all end in "_contacts"
    contact_filters = [key for key in kwargs.keys() if key.endswith("_contacts")]
    # obtain the contact filters, make a dict of target_name: int(contacts)
    contact_filters_dict = {key[:-9]: int(kwargs[key]) for key in contact_filters}
    # loop over inputs
    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        chB_length = pose.chain_end(2) - pose.chain_end(1)
        # loop over targets to thread onto inputs
        for target_name, target_seq in target_dict.items():
            if len(target_seq) > chB_length:
                # thread varius frames of the target onto the entire input
                to_thread = [
                    target_seq[x:y]
                    for x, y in combinations(range(len(target_seq) + 1), r=2)
                    if len(target_seq[x:y]) == chB_length
                ]
                thread_starts = [pose.chain_begin(2)]
                thread_ends = [pose.chain_end(2)]
            # thread the entire target onto various frames of the input
            else:
                to_thread = [target_seq]
                thread_starts = [
                    i
                    for i in range(
                        int(pose.chain_begin(2)),
                        int(pose.chain_end(2)) - len(target_seq) + 2,
                    )
                ]
                thread_ends = [i + len(target_seq) - 1 for i in thread_starts]

            # make movers for the threading protocol
            sc = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
            sc.chain_order("12")
            chA_only = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
            chA_only.chain_order("1")
            stm = pyrosetta.rosetta.protocols.simple_moves.SimpleThreadingMover()
            trim = pyrosetta.rosetta.protocols.grafting.simple_movers.KeepRegionMover()
            trim_cterm = (
                pyrosetta.rosetta.protocols.grafting.simple_movers.DeleteRegionMover()
            )
            # make residue selectors
            apolar = ResidueNameSelector()
            apolar.set_residue_name3("ALA,PHE,ILE,LEU,MET,VAL,TRP,TYR")
            chA_selector = ChainSelector(1)
            chB_selector = ChainSelector(2)
            interface_selector = interface_between_selectors(chA_selector, chB_selector)
            interface_chB = AndResidueSelector(chB_selector, interface_selector)
            interface_chB_apolar = AndResidueSelector(interface_chB, apolar)
            best_poses = []
            best_int_count = 0
            # loop over frames to thread x thread starts
            for seq_to_thread in to_thread:
                for i, start in enumerate(thread_starts):
                    end = thread_ends[i]
                    # create a threading mover
                    stm.set_sequence(seq_to_thread, start)
                    threaded_pose = pose.clone()
                    stm.apply(threaded_pose)
                    # clean off trailing ends
                    chA, chB = threaded_pose.clone(), threaded_pose.clone()
                    chA_only.apply(chA)
                    if i < len(thread_starts) - 1:
                        trim.region(str(start), str(end))
                        trim.apply(chB)
                    else:
                        # keepregionmover has a bug with c terminal regions
                        trim_cterm.region(str(chB.chain_begin(1)), str(start - 1))
                        trim_cterm.apply(chB)
                    chA.append_pose_by_jump(chB, chA.num_jump() + 1)
                    sc.apply(chA)
                    # find all residues in the interface that are apolar and on chB
                    list_of_apolar_in_int = list(interface_chB_apolar.apply(chA))
                    # convert to a count by summing the True values
                    count_of_apolar_in_int = sum(list_of_apolar_in_int)
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(
                        chA, "count_apolar", count_of_apolar_in_int
                    )
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(
                        chA, "kept_start", start
                    )
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(chA, "kept_end", end)
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(
                        chA, "threaded_seq", seq_to_thread
                    )
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(
                        chA, "target_seq", target_seq
                    )
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(
                        chA, "target_name", target_name
                    )
                    prefix, suffix = (
                        target_seq.split(seq_to_thread)[0],
                        target_seq.split(seq_to_thread)[-1],
                    )
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(chA, "prefix", prefix)
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(chA, "suffix", suffix)
                    # if the count is better than the best count, start a new list
                    if count_of_apolar_in_int > best_int_count:
                        best_int_count = count_of_apolar_in_int
                        best_poses = [chA]
                    # if the count is the same as the best count, add to the list
                    elif count_of_apolar_in_int == best_int_count:
                        best_poses.append(chA)
                    # otherwise, continue
                    else:
                        continue
            # yield all the best poses after updating the scores
            for best_pose in best_poses:
                # check if there is a contact filter for the pose target_name
                if best_pose.scores["target_name"] in contact_filters_dict.keys():
                    # check if the pose passes the contact filter
                    if (
                        best_pose.scores["count_apolar"]
                        > contact_filters_dict[best_pose.scores["target_name"]]
                    ):
                        pass
                    else:
                        continue
                else:
                    pass
                final_scores = {**scores, **dict(best_pose.scores)}
                for key, value in final_scores.items():
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(
                        best_pose, key, str(value)
                    )
                yield best_pose


@requires_init
def detail(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param packed_pose_in: PackedPose object.
    :param kwargs: kwargs such as "pdb_path".
    :return: Iterator of PackedPose objects with terminal repeats designed with MPNN.
    """

    import sys
    from operator import gt, lt
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        NeighborhoodResidueSelector,
        ResidueIndexSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.folding import (
        SuperfoldRunner,
        generate_decoys_from_pose,
    )
    from crispy_shifty.protocols.mpnn import MPNNDesign, dict_to_fasta, fasta_to_dict
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        poses = path_to_pose_or_ppose(
            path=kwargs["pdb_path"], cluster_scores=True, pack_result=False
        )
    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        # fix chain and residue labels and PDBInfo
        sc = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        sc.chain_order("".join(str(x) for x in range(1, pose.num_chains() + 1)))
        sc.apply(pose)
        # get number of repeats that were extended by
        try:
            resis = kwargs["resis"]
        except KeyError:
            resis = None
        # see if a neighborhood distance is in kwargs
        try:
            neighborhood_distance = int(kwargs["neighborhood_distance"])
        except KeyError:
            neighborhood_distance = 6
        # if resis is not None, use it to select the residues to design
        if resis is not None:
            resis_sel = ResidueIndexSelector(resis)
            design_sel = AndResidueSelector(
                ChainSelector(1),
                NeighborhoodResidueSelector(resis_sel, neighborhood_distance),
            )
        else:
            design_sel = ChainSelector(1)
        print_timestamp("Redesigning with MPNN", start_time)
        # construct the MPNNDesign object
        mpnn_design = MPNNDesign(
            design_selector=design_sel,
            omit_AAs="CX",
            temperature=0.05,
            **kwargs,
        )
        # design the pose
        mpnn_design.apply(pose)
        print_timestamp("MPNN design complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        print_timestamp("Filtering sequences with AF2", start_time)
        # make a temporary fasta dict from the remaining mpnn_seq scores
        tmp_fasta_dict = {k: v for k, v in pose.scores.items() if "mpnn_seq" in k}
        # fix the fasta by splitting on chainbreaks '/' and rejoining the first two
        tmp_fasta_dict = {
            tag: "/".join(seq.split("/")[0:2]) for tag, seq in tmp_fasta_dict.items()
        }
        print_timestamp("Setting up for AF2", start_time)
        # setup with dummy fasta path
        runner = SuperfoldRunner(
            pose=pose, fasta_path="dummy", load_decoys=True, **kwargs
        )
        runner.setup_runner()
        # initial_guess, reference_pdb both are the tmp.pdb
        initial_guess = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        reference_pdb = initial_guess
        flag_update = {
            "--initial_guess": initial_guess,
            "--reference_pdb": reference_pdb,
        }
        # now we have to point to the right fasta file
        new_fasta_path = str(Path(runner.get_tmpdir()) / "tmp.fa")
        dict_to_fasta(tmp_fasta_dict, new_fasta_path)
        runner.set_fasta_path(new_fasta_path)
        runner.override_input_file(new_fasta_path)
        runner.update_flags(flag_update)
        runner.update_command()
        print_timestamp("Running AF2", start_time)
        runner.apply(pose)
        print_timestamp("AF2 complete, updating pose datacache", start_time)
        # check kwargs for filtering instructions
        if "plddt_cutoff" in kwargs:
            plddt_cutoff = kwargs["plddt_cutoff"]
        else:
            plddt_cutoff = 92.0
        if "rmsd_cutoff" in kwargs:
            rmsd_cutoff = kwargs["rmsd_cutoff"]
        else:
            rmsd_cutoff = 1.5
        filter_dict = {
            "mean_plddt": (gt, plddt_cutoff),
            "rmsd_to_reference": (lt, rmsd_cutoff),
        }
        # setup prefix, rank_on
        rank_on = "rmsd_to_reference"
        prefix = "mpnn_seq"
        print_timestamp("Adding passing sequences back into pose", start_time)
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            generate_prediction_decoys=True,
            label_first=False,
            prefix=prefix,
            rank_on=rank_on,
        ):
            packed_decoy = io.to_packed(decoy)
            yield packed_decoy

@requires_init
def filter_binder(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be filtered.
    :param: kwargs: keyword arguments to be passed to this function.
    :return: an iterator of PackedPose objects.
    No SAP for now. 
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        TrueResidueSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import gen_scorefxn, gen_task_factory, pack_rotamers, score_cms, score_ddg
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

    # make selectors
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    # make stuff for repacker
    scorefxn = gen_scorefxn(cartesian=False, res_type_constraint=False, hbonds=False, weights="ref2015")
    task_factory = gen_task_factory(
        design_sel=None,
        pack_sel=TrueResidueSelector(),
    )
    # make secretion_predictor
    xml = """<FILTERS>\n<SecretionPredictionFilter name="insertion_dG" />\n</FILTERS>"""
    obj = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml)
    secretion_predictor = obj.get_filter("insertion_dG")
    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        # repack pose
        print_timestamp("Repacking pose", start_time)
        pack_rotamers(
            pose=pose,
            scorefxn=scorefxn,
            task_factory=task_factory,
        )
        print_timestamp(f"Filtering", start_time)
        # get cms
        cms = score_cms(pose=pose, sel_1=chA, sel_2=chB)
        # get ddG
        ddg = score_ddg(pose=pose)
        # get membrane insertion dG
        insertion_dG = secretion_predictor.report_sm(pose)
        pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "insertion_dG", insertion_dG)
        print_timestamp("Filtering complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        scores["chA_seq"] = pose.split_by_chain(1).sequence()
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # generate the filtered pose
        ppose = io.to_packed(pose)
        yield ppose

@requires_init
def sc_binder(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be filtered.
    :param: kwargs: keyword arguments to be passed to this function.
    :return: an iterator of PackedPose objects.
    No SAP for now. 
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        TrueResidueSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import gen_scorefxn, gen_task_factory, pack_rotamers, score_cms, score_ddg, score_sc
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

    ids = list(pdb_path.split(" "))

    # make selectors
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    # make stuff for repacker
    scorefxn = gen_scorefxn(cartesian=False, res_type_constraint=False, hbonds=False, weights="ref2015")
    task_factory = gen_task_factory(
        design_sel=None,
        pack_sel=TrueResidueSelector(),
    )
    # make secretion_predictor
    xml = """<FILTERS>\n<SecretionPredictionFilter name="insertion_dG" />\n</FILTERS>"""
    obj = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml)
    secretion_predictor = obj.get_filter("insertion_dG")
    for i, pose in enumerate(poses):
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        # repack pose
        print_timestamp("Repacking pose", start_time)
        # pack_rotamers(
        #     pose=pose,
        #     scorefxn=scorefxn,
        #     task_factory=task_factory,
        # )
        print_timestamp(f"Filtering", start_time)
        # get cms
        cms = score_cms(pose=pose, sel_1=chA, sel_2=chB)
        # get ddG
        ddg = score_ddg(pose=pose)
        # get ddG
        sc = score_sc(pose=pose, sel_1=chA, sel_2=chB)
        # get membrane insertion dG
        insertion_dG = secretion_predictor.report_sm(pose)
        # update the scores dict
        scores = dict(pose.scores)
        scores["id"] = ids[i]
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # generate the filtered pose
        ppose = io.to_packed(pose)
        yield ppose

