# Python standard library
from typing import Iterator, List, Optional, Tuple, Union

from pyrosetta.distributed import requires_init

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.scoring import ScoreFunction
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports


def almost_linkres(
    pose: Pose,
    movemap: MoveMap,
    residue_selectors: Union[List[ResidueSelector], Tuple[ResidueSelector]],
    scorefxn: ScoreFunction,
    task_factory: TaskFactory,
    repeats: int = 1,
) -> None:
    """
    :param: pose: The pose to be designed.
    :param: movemap: The movemap to be used for design.
    :param: residue_selectors: The residue selectors to be used for linking.
    :param: scorefxn: The score function to be used for scoring.
    :param: task_factory: The task factory to be used for design.
    :param: repeats: The number of times to repeat the design.
    :return: None

    This function does fast design using a linkres-style approach.
    It requires at minimum a pose, movemap, scorefxn, and task_factory.
    The pose will be modified in place with fast_design, and the movemap and scorefxn
    will be passed directly to fast_design. The task_factory will have a sequence
    symmetry taskop added before it will be passed to fast_design. The residue_selectors
    will be used to determine which residues to pseudosymmetrize, and need to specify
    equal numbers of residues for each selector.
    """
    import sys
    from pathlib import Path

    import pyrosetta

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.design import fast_design

    # make a dict of the residue selectors as index strings
    index_selectors = {}
    for i, selector in enumerate(residue_selectors):
        index_selectors[f"sel_{i}"] = ",".join(
            [str(j) for j, pos in list(enumerate(selector.apply(pose), start=1)) if pos]
        )
    pre_xml_string = """
    <RESIDUE_SELECTORS>
        {index_selectors_str}
    </RESIDUE_SELECTORS>
    <TASKOPERATIONS>
        <KeepSequenceSymmetry name="linkres_op" setting="true"/>
    </TASKOPERATIONS>
    <MOVERS>
        <SetupForSequenceSymmetryMover name="almost_linkres" sequence_symmetry_behaviour="linkres_op" >
            <SequenceSymmetry residue_selectors="{selector_keys}" />
        </SetupForSequenceSymmetryMover>
    </MOVERS>
    """
    index_selectors_str = "\n\t\t".join(
        [
            f"""<Index name="{key}" resnums="{value}" />"""
            for key, value in index_selectors.items()
        ]
    )
    # autogenerate an xml string
    xml_string = pre_xml_string.format(
        index_selectors_str=index_selectors_str,
        selector_keys=",".join(index_selectors.keys()),
    )
    # setup the xml_object
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        xml_string
    )
    # get the taskop from the xml_object
    linkres_op = objs.get_task_operation("linkres_op")
    # push back the taskop to the task factory
    task_factory.push_back(linkres_op)
    # get the mover from the xml_object
    pre_linkres = objs.get_mover("almost_linkres")
    # apply the mover to the pose
    pre_linkres.apply(pose)
    # run fast_design with the pose, movemap, scorefxn, task_factory
    fast_design(
        pose=pose,
        movemap=movemap,
        scorefxn=scorefxn,
        task_factory=task_factory,
        repeats=repeats,
    )
    return


@requires_init
def two_state_design_paired_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a packed pose to use as a starting point for multistate
    design. If None, a pose will be generated from the input pdb_path.
    :param: kwargs: keyword arguments for almost_linkres.
    Needs `-corrections:beta_nov16 true` in init.
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        OrResidueSelector,
        ResidueIndexSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        add_metadata_to_pose,
        fast_design,
        gen_movemap,
        gen_score_filter,
        gen_std_layer_design,
        gen_task_factory,
        interface_among_chains,
        score_per_res,
        score_ss_sc,
    )
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()
    # setup scorefxns
    clean_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
    )
    print_timestamp("Generated score functions", start_time=start_time)
    # setup movemap
    flexbb_mm = gen_movemap(jump=True, chi=True, bb=True)
    print_timestamp("Generated movemaps", start_time=start_time)
    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )
    skip_resi_subtype_check = False
    if "skip_resi_subtype_check" in kwargs:
        if kwargs["skip_resi_subtype_check"].lower() == "true":
            skip_resi_subtype_check = True
    # gogogo
    for pose in poses:
        start_time = time()
        # get the scores from the pose
        scores = dict(pose.scores)
        # make a list to append designed poses to
        designed_poses = []
        # for the neighborhood residue selector
        pose.update_residue_neighbors()
        # get the chains
        chA, chB, chC = (ChainSelector(i) for i in range(1, 4))
        # get the bound interface
        interface_sel = interface_among_chains(chain_list=[1, 2], vector_mode=True)
        # get the chB interface
        chB_interface_sel = AndResidueSelector(chB, interface_sel)
        offset = pose.chain_end(2)

        # fix cysteines that are present in one state but not the other by mutating them to alanine
        difference_cys_indices = []
        for i in range(1, pose.chain_end(1) + 1):
            n1 = str(pose.residue(i).name1())
            n2 = str(pose.residue(i + offset).name1())
            if n1 == "C" and n2 != "C":
                difference_cys_indices.append(i)
            elif n1 != "C" and n2 == "C":
                difference_cys_indices.append(i + offset)
        if difference_cys_indices:
            diff_cys_sel = ResidueIndexSelector(
                ",".join(str(i) for i in sorted(difference_cys_indices))
            )
            mr = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
            mr.set_break_disulfide_bonds(True)
            mr.set_preserve_atom_coords(True)
            mr.set_res_name("ALA")
            mr.set_selector(diff_cys_sel)
            mr.apply(pose)

        # get any residues that differ between chA and chC - starts as a list of tuples
        if skip_resi_subtype_check:
            difference_indices = [
                (i, i + offset)
                for i in range(1, pose.chain_end(1) + 1)
                if pose.residue(i).name1() != pose.residue(i + offset).name1()
            ]
        else:
            difference_indices = [
                (i, i + offset)
                for i in range(1, pose.chain_end(1) + 1)
                if pose.residue(i).name() != pose.residue(i + offset).name()
            ]
        # flatten the list of tuples into a sorted list of indices
        difference_indices = sorted(sum(difference_indices, ()))
        # make a residue selector for the difference indices
        difference_sel = ResidueIndexSelector(",".join(map(str, difference_indices)))
        # use OrResidueSelector to combine the two
        design_sel = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(
            chB_interface_sel, difference_sel
        )
        # we want to design the peptide (chB) interface + anything that differs between the states
        print_timestamp("Generated selectors", start_time=start_time)
        # we need to add an alanine to all layers of the default list
        layer_aas_list = [
            "ADNSTP",  # helix_cap
            "AFILVWYNQSTHP",  # core AND helix_start
            "AFILVWM",  # core AND helix
            "AFILVWY",  # core AND sheet
            "AFGILPVWYSM",  # core AND loop
            "ADEHIKLNPQRSTVWY",  # boundary AND helix_start
            "ADEHIKLNQRSTVWYM",  # boundary AND helix
            "ADEFHIKLNQRSTVWY",  # boundary AND sheet
            "ADEFGHIKLNPQRSTVWY",  # boundary AND loop
            "ADEHKPQR",  # surface AND helix_start
            "AEHKQR",  # surface AND helix
            "AEHKNQRST",  # surface AND sheet
            "ADEGHKNPQRST",  # surface AND loop
        ]
        layer_design = gen_std_layer_design(layer_aas_list=layer_aas_list)
        task_factory_1 = gen_task_factory(
            design_sel=design_sel,
            pack_nbhd=True,
            extra_rotamers_level=1,
            limit_arochi=True,
            prune_buns=True,
            upweight_ppi=True,
            restrict_pro_gly=True,
            precompute_ig=True,
            ifcl=True,
            layer_design=layer_design,
        )
        print_timestamp(
            "Generated interface design task factory with upweighted interface",
            start_time=start_time,
        )
        # setup the linked selectors
        residue_selectors = chA, chC
        print_timestamp(
            "Starting 1 round of flexbb msd with upweighted interface",
            start_time=start_time,
        )
        almost_linkres(
            pose=pose,
            movemap=flexbb_mm,
            residue_selectors=residue_selectors,
            scorefxn=design_sfxn,
            task_factory=task_factory_1,
            repeats=1,
        )
        add_metadata_to_pose(pose, "interface", "upweight")
        designed_poses.append(pose.clone())
        print_timestamp(
            "Starting 1 round of flexbb design and non-upweighted interface",
            start_time=start_time,
        )
        task_factory_2 = gen_task_factory(
            design_sel=design_sel,
            pack_nbhd=True,
            extra_rotamers_level=1,
            limit_arochi=True,
            prune_buns=True,
            upweight_ppi=False,
            restrict_pro_gly=True,
            precompute_ig=True,
            ifcl=True,
            layer_design=layer_design,
        )
        print_timestamp(
            "Generated interface design task factory with non-upweighted interface",
            start_time=start_time,
        )
        almost_linkres(
            pose=pose,
            movemap=flexbb_mm,
            residue_selectors=residue_selectors,
            scorefxn=design_sfxn,
            task_factory=task_factory_2,
            repeats=1,
        )
        add_metadata_to_pose(pose, "interface", "normal")
        designed_poses.append(pose.clone())
        for pose in designed_poses:
            print_timestamp("Scoring...", start_time=start_time)
            score_per_res(pose, clean_sfxn)
            score_ss_sc(pose)
            score_filter = gen_score_filter(clean_sfxn)
            add_metadata_to_pose(pose, "path_in", pdb_path)
            end_time = time()
            total_time = end_time - start_time
            print_timestamp(
                f"Total time: {total_time:.2f} seconds", start_time=start_time
            )
            add_metadata_to_pose(pose, "time", total_time)
            scores.update(pose.scores)
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def two_state_design_paired_state_one_sided(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a packed pose to use as a starting point for multistate
    design. If None, a pose will be generated from the input pdb_path.
    :param: kwargs: keyword arguments for almost_linkres.
    Needs `-corrections:beta_nov16 true` in init.
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        OrResidueSelector,
        ResidueIndexSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        add_metadata_to_pose,
        fast_design,
        gen_movemap,
        gen_score_filter,
        gen_std_layer_design,
        gen_task_factory,
        interface_among_chains,
        score_per_res,
        score_ss_sc,
    )
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()
    # setup scorefxns
    clean_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    if "sfxn_wts" in kwargs.keys():        
        clean_sfxn = pyrosetta.create_score_function(kwargs["sfxn_wts"])
        design_sfxn = pyrosetta.create_score_function(kwargs["sfxn_wts"])
    design_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
    )
    print_timestamp("Generated score functions", start_time=start_time)
    # setup movemap
    flexbb_mm = gen_movemap(jump=True, chi=True, bb=True)
    print_timestamp("Generated movemaps", start_time=start_time)
    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )
    skip_resi_subtype_check = False
    if "skip_resi_subtype_check" in kwargs:
        if kwargs["skip_resi_subtype_check"].lower() == "true":
            skip_resi_subtype_check = True
    # gogogo
    for pose in poses:
        start_time = time()
        # get the scores from the pose
        scores = dict(pose.scores)
        # make a list to append designed poses to
        designed_poses = []
        # for the neighborhood residue selector
        pose.update_residue_neighbors()
        # get the chains
        chA, chB, chC = (ChainSelector(i) for i in range(1, 4))
        # get the bound interface
        interface_sel = interface_among_chains(chain_list=[1, 2], vector_mode=True)
        # get the chB interface
        chB_interface_sel = AndResidueSelector(chB, interface_sel)
        offset = pose.chain_end(2)

        # fix cysteines that are present in one state but not the other by mutating them to alanine
        difference_cys_indices = []
        for i in range(1, pose.chain_end(1) + 1):
            n1 = str(pose.residue(i).name1())
            n2 = str(pose.residue(i + offset).name1())
            if n1 == "C" and n2 != "C":
                difference_cys_indices.append(i)
            elif n1 != "C" and n2 == "C":
                difference_cys_indices.append(i + offset)
        if difference_cys_indices:
            diff_cys_sel = ResidueIndexSelector(
                ",".join(str(i) for i in sorted(difference_cys_indices))
            )
            mr = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
            mr.set_break_disulfide_bonds(True)
            mr.set_preserve_atom_coords(True)
            mr.set_res_name("ALA")
            mr.set_selector(diff_cys_sel)
            mr.apply(pose)

        # get any residues that differ between chA and chC - starts as a list of tuples
        if skip_resi_subtype_check:
            difference_indices = [
                (i, i + offset)
                for i in range(1, pose.chain_end(1) + 1)
                if pose.residue(i).name1() != pose.residue(i + offset).name1()
            ]
        else:
            difference_indices = [
                (i, i + offset)
                for i in range(1, pose.chain_end(1) + 1)
                if pose.residue(i).name() != pose.residue(i + offset).name()
            ]
        # flatten the list of tuples into a sorted list of indices
        difference_indices = sorted(sum(difference_indices, ()))
        # make a residue selector for the difference indices
        difference_sel = ResidueIndexSelector(",".join(map(str, difference_indices)))
        # use OrResidueSelector to combine the two
        design_sel = difference_sel
        # we want to design anything on ChA/ChC that differs between the states
        print_timestamp("Generated selectors", start_time=start_time)
        # we need to add an alanine to all layers of the default list
        layer_aas_list = [
            "ADNSTP",  # helix_cap
            "AFILVWYNQSTHP",  # core AND helix_start
            "AFILVWM",  # core AND helix
            "AFILVWY",  # core AND sheet
            "AFGILPVWYSM",  # core AND loop
            "ADEHIKLNPQRSTVWY",  # boundary AND helix_start
            "ADEHIKLNQRSTVWYM",  # boundary AND helix
            "ADEFHIKLNQRSTVWY",  # boundary AND sheet
            "ADEFGHIKLNPQRSTVWY",  # boundary AND loop
            "ADEHKPQR",  # surface AND helix_start
            "AEHKQR",  # surface AND helix
            "AEHKNQRST",  # surface AND sheet
            "ADEGHKNPQRST",  # surface AND loop
        ]
        layer_design = gen_std_layer_design(layer_aas_list=layer_aas_list)
        task_factory_1 = gen_task_factory(
            design_sel=design_sel,
            pack_nbhd=True,
            extra_rotamers_level=1,
            limit_arochi=True,
            prune_buns=True,
            upweight_ppi=True,
            restrict_pro_gly=True,
            precompute_ig=True,
            ifcl=True,
            layer_design=layer_design,
        )
        print_timestamp(
            "Generated interface design task factory with upweighted interface",
            start_time=start_time,
        )
        # setup the linked selectors
        residue_selectors = chA, chC
        print_timestamp(
            "Starting 1 round of flexbb msd with upweighted interface",
            start_time=start_time,
        )
        almost_linkres(
            pose=pose,
            movemap=flexbb_mm,
            residue_selectors=residue_selectors,
            scorefxn=design_sfxn,
            task_factory=task_factory_1,
            repeats=1,
        )
        add_metadata_to_pose(pose, "interface", "upweight")
        designed_poses.append(pose.clone())
        print_timestamp(
            "Starting 1 round of flexbb design and non-upweighted interface",
            start_time=start_time,
        )
        task_factory_2 = gen_task_factory(
            design_sel=design_sel,
            pack_nbhd=True,
            extra_rotamers_level=1,
            limit_arochi=True,
            prune_buns=True,
            upweight_ppi=False,
            restrict_pro_gly=True,
            precompute_ig=True,
            ifcl=True,
            layer_design=layer_design,
        )
        print_timestamp(
            "Generated interface design task factory with non-upweighted interface",
            start_time=start_time,
        )
        almost_linkres(
            pose=pose,
            movemap=flexbb_mm,
            residue_selectors=residue_selectors,
            scorefxn=design_sfxn,
            task_factory=task_factory_2,
            repeats=1,
        )
        add_metadata_to_pose(pose, "interface", "normal")
        designed_poses.append(pose.clone())
        for pose in designed_poses:
            print_timestamp("Scoring...", start_time=start_time)
            score_per_res(pose, clean_sfxn)
            score_ss_sc(pose)
            score_filter = gen_score_filter(clean_sfxn)
            add_metadata_to_pose(pose, "path_in", pdb_path)
            end_time = time()
            total_time = end_time - start_time
            print_timestamp(
                f"Total time: {total_time:.2f} seconds", start_time=start_time
            )
            add_metadata_to_pose(pose, "time", total_time)
            scores.update(pose.scores)
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def filter_paired_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a packed pose to filter. If None, a pose will be generated 
    from the input pdb_path.
    :param: kwargs: keyword arguments for filtering.
    Needs `-corrections:beta_nov16 true` and 
    `-indexed_structure_store:fragment_store \
    /net/databases/VALL_clustered/connect_chains/ss_grouped_vall_helix_shortLoop.h5` in
    init statement.
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        FalseResidueSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        gen_std_layer_design,
        gen_task_factory,
        pack_rotamers,
        score_cms,
        score_ddg,
        score_per_res,
        score_SAP,
        score_wnm_all,
        score_wnm_helix,
    )
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()
    # setup scorefxns
    clean_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    print_timestamp("Generated score functions", start_time=start_time)
    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )
    # make a repack only task factory
    task_factory = gen_task_factory(
        design_sel=FalseResidueSelector(),
        pack_nbhd=False,
        pack_nondesignable=True,
        extra_rotamers_level=2,
        limit_arochi=True,
        prune_buns=True,
        upweight_ppi=False,
        restrict_pro_gly=True,
        precompute_ig=True,
        ifcl=True,
        layer_design=gen_std_layer_design(),
    )
    # gogogo
    for pose in poses:
        start_time = time()
        # get the scores from the pose
        scores = dict(pose.scores)
        # for the neighborhood residue selector
        pose.update_residue_neighbors()
        print_timestamp("Settling Y and scoring...", start_time=start_time)
        # pack the rotamers to get rid of rosetta clashes
        pack_rotamers(pose=pose, task_factory=task_factory, scorefxn=clean_sfxn)
        # get SAP
        # TODO this is scoring the sap of the entire pose, not just chains A and B
        Y_sap = score_SAP(pose, name="Y_sap")
        # get cms
        chA, chB, chC = (ChainSelector(i) for i in range(1, 4))
        Y_cms = score_cms(pose=pose, sel_1=chA, sel_2=chB, name="Y_cms")
        # get ddg
        Y_ddg = score_ddg(pose=pose, name="Y_ddg")
        # get total score and score_per_res
        # TODO this is scoring the sap of the entire pose, not just chains A and B
        Y_total_score, Y_score_per_res = score_per_res(pose=pose, scorefxn=clean_sfxn)
        # update the scores dict with the new scores
        Y_scores = {
            "Y_sap": Y_sap,
            "Y_cms": Y_cms,
            "Y_ddg": Y_ddg,
            "Y_total_score": Y_total_score,
            "Y_score_per_res": Y_score_per_res,
        }
        scores.update(Y_scores)

        sw = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        sw.chain_order("1")

        for solo_chain, chain_id in zip(list(pose.split_by_chain()), "ABC"):
            print_timestamp(f"Scoring {chain_id}...", start_time=start_time)
            # currently commented out because it should be fine to use the pose straight out of split_by_chain, and append_pose_to_pose somehow messes up disulfides
            # make a solo copy of the chain
            # solo_chain = Pose()
            # pyrosetta.rosetta.core.pose.append_pose_to_pose(
            #     solo_chain,
            #     chain,
            #     new_chain=True,
            # )
            sw.apply(solo_chain)
            # repack the chain
            pack_rotamers(
                pose=solo_chain, task_factory=task_factory, scorefxn=clean_sfxn
            )
            # get SAP
            sap = score_SAP(pose=solo_chain, name=f"{chain_id}_sap")
            # get total score and score_per_res
            chain_total_score, chain_score_per_res = score_per_res(
                solo_chain, clean_sfxn, name=f"{chain_id}_score"
            )
            # get wnm_all
            wnm_all = score_wnm_all(solo_chain)[0]
            # get wnm_helix
            wnm_helix = score_wnm_helix(solo_chain, name=f"{chain_id}_wnm_helix")
            # update the scores dict with the new scores
            final_sequence = solo_chain.sequence()
            chain_scores = {
                f"{chain_id}_final_seq": final_sequence,
                f"{chain_id}_sap": sap,
                f"{chain_id}_total_score": chain_total_score,
                f"{chain_id}_score_per_res": chain_score_per_res,
                f"{chain_id}_wnm_all": wnm_all,
                f"{chain_id}_wnm_helix": wnm_helix,
            }
            scores.update(chain_scores)

        end_time = time()
        total_time = end_time - start_time
        print_timestamp(f"Total time: {total_time:.2f} seconds", start_time=start_time)
        # clear the pose scores
        pyrosetta.rosetta.core.pose.clearPoseExtraScores(pose)
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        ppose = io.to_packed(pose)
        yield ppose
