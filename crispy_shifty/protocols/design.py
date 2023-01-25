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
from pyrosetta.rosetta.protocols.filters import Filter

# Custom library imports

beta_nov16_terms = [
    "dslf_fa13",
    "fa_atr",
    "fa_dun",
    "fa_dun_dev",
    "fa_dun_rot",
    "fa_dun_semi",
    "fa_elec",
    "fa_dun_dev",
    "fa_dun_rot",
    "fa_dun_semi",
    "fa_elec",
    "fa_intra_atr_xover4",
    "fa_intra_elec",
    "fa_intra_rep",
    "fa_intra_rep_xover4",
    "fa_intra_sol_xover4",
    "fa_rep",
    "fa_sol",
    "hbond_bb_sc",
    "hbond_lr_bb",
    "hbond_sc",
    "hbond_sr_bb",
    "hxl_tors",
    "lk_ball",
    "lk_ball_bridge",
    "lk_ball_bridge_uncpl",
    "lk_ball_iso",
    "lk_ball_wtd",
    "omega",
    "p_aa_pp",
    "pro_close",
    "rama_prepro",
    "ref",
    "res_type_constraint",
    "yhh_planarity",
]


def clear_terms_from_scores(pose: Pose, terms: Optional[List[str]] = None) -> None:
    """
    :param: pose: The pose to clear the terms from.
    :param: terms: The terms to clear from the pose.
    :return: None.
    Clears beta nov16 terms from the pose by default, or the given terms if provided.
    """
    import sys
    from pathlib import Path

    import pyrosetta

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.design import beta_nov16_terms

    if terms is None:
        terms = beta_nov16_terms
    else:
        pass

    for term in terms:
        pyrosetta.rosetta.core.pose.clearPoseExtraScore(pose, term)
    return


def add_metadata_to_pose(
    pose: Pose, key: str, metadata: Union[int, float, str]
) -> None:
    """
    :param pose: Pose to add metadata to
    :param metadata: Arbitrary metadata to add to the pose
    :return: None
    Adds arbitrary metadata to the pose datacache as a score.
    """
    import pyrosetta

    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, str(metadata))
    return


def selector_union(
    selectors: List[ResidueSelector], invert: bool = False
) -> ResidueSelector:
    """
    :param: selectors: List[ResidueSelector], the selectors to union.
    :param: invert: bool, whether to invert the union.
    :return: ResidueSelector, the union of the selectors.
    Returns the union of the given selectors.
    """
    from pyrosetta.rosetta.core.select.residue_selector import (
        OrResidueSelector,
        NotResidueSelector,
    )

    selector = OrResidueSelector()
    for sel in selectors:
        selector.add_residue_selector(sel)
    if invert:
        return NotResidueSelector(selector)
    else:
        return selector


def selector_intersection(
    selectors: List[ResidueSelector], invert: bool = False
) -> ResidueSelector:
    """
    :param: selectors: List[ResidueSelector], the selectors to union.
    :param: invert: bool, whether to invert the union.
    :return: ResidueSelector, the intersection of the selectors.
    Returns the intersection of the given selectors.
    """
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        NotResidueSelector,
    )

    selector = AndResidueSelector()
    for sel in selectors:
        selector.add_residue_selector(sel)
    if invert:
        return NotResidueSelector(selector)
    else:
        return selector


def interface_between_selectors(
    sel_1: ResidueSelector, sel_2: ResidueSelector, vector_mode: bool = False
) -> ResidueSelector:
    """
    :param: sel_1: ResidueSelector for the first half of the interface.
    :param: sel_2: ResidueSelector for the second half of the interface.
    :param: vector_mode: If true, use vectors of Cb atoms in addition to distance.
    :return: ResidueSelector that selects the interface between the two selectors.
    Returns a selector that selects the interface between two selectors.
    """
    if vector_mode:
        from pyrosetta.rosetta.core.select.residue_selector import (
            InterGroupInterfaceByVectorSelector,
        )

        int_sel = InterGroupInterfaceByVectorSelector(sel_1, sel_2)
        # tuned for TJ DHRs
        int_sel.nearby_atom_cut(4)
        int_sel.vector_dist_cut(6.5)
        int_sel.cb_dist_cut(8.5)
        return int_sel
    else:
        from pyrosetta.rosetta.core.select.residue_selector import (
            AndResidueSelector,
            NeighborhoodResidueSelector,
        )

        sel_1_nbhd = NeighborhoodResidueSelector(sel_1, 8, True)
        sel_2_nbhd = NeighborhoodResidueSelector(sel_2, 8, True)
        return AndResidueSelector(sel_1_nbhd, sel_2_nbhd)


def interface_among_chains(
    chain_list: list, vector_mode: bool = False
) -> ResidueSelector:
    """
    :param: chain_list: List of chains to select the interface between.
    :param: vector_mode: If true, use vectors of Cb atoms in addition to distance.
    :return: ResidueSelector that selects the interface between the chains.
    Returns a selector that selects the interface between the given chains of a pose.
    """
    from itertools import combinations

    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        OrResidueSelector,
    )

    int_sel = OrResidueSelector()
    for chain_1, chain_2 in combinations(chain_list, 2):
        sel_1 = ChainSelector(chain_1)
        sel_2 = ChainSelector(chain_2)
        pair_int_sel = interface_between_selectors(sel_1, sel_2, vector_mode)
        int_sel.add_residue_selector(pair_int_sel)

    return int_sel


def gen_std_layer_design(layer_aas_list: list = None) -> dict:
    """
    :param: layer_aas_list: List of amino acids to include in each layer. Must be of
    length 13.
    :return: Dictionary of layer design definitions.
    Returns a dictionary of layer design definitions.
    """

    from itertools import product

    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        LayerSelector,
        NotResidueSelector,
        PrimarySequenceNeighborhoodSelector,
        SecondaryStructureSelector,
    )

    layer_names = [
        "helix_cap",
        "core AND helix_start",
        "core AND helix",
        "core AND sheet",
        "core AND loop",
        "boundary AND helix_start",
        "boundary AND helix",
        "boundary AND sheet",
        "boundary AND loop",
        "surface AND helix_start",
        "surface AND helix",
        "surface AND sheet",
        "surface AND loop",
    ]

    if layer_aas_list is None:  # set default layer design
        layer_aas_list = [
            "DNSTP",  # helix_cap
            "AFILVWYNQSTHP",  # core AND helix_start
            "AFILVWM",  # core AND helix
            "FILVWY",  # core AND sheet
            "AFGILPVWYSM",  # core AND loop
            "ADEHIKLNPQRSTVWY",  # boundary AND helix_start
            "ADEHIKLNQRSTVWYM",  # boundary AND helix
            "DEFHIKLNQRSTVWY",  # boundary AND sheet
            "ADEFGHIKLNPQRSTVWY",  # boundary AND loop
            "DEHKPQR",  # surface AND helix_start
            "EHKQR",  # surface AND helix
            "EHKNQRST",  # surface AND sheet
            "DEGHKNPQRST",  # surface AND loop
        ]
    assert len(layer_aas_list) == 13

    layer_sels = []  # core, boundary, surface
    for layer in ["core", "bdry", "surf"]:
        layer_sel = LayerSelector()
        # layer_sel.set_layers(i==0, i==1, i==2) # 1-liner when iterating through a range, but less easy to read
        if layer == "core":
            layer_sel.set_layers(True, False, False)
        elif layer == "bdry":
            layer_sel.set_layers(False, True, False)
        elif layer == "surf":
            layer_sel.set_layers(False, False, True)
        layer_sel.set_use_sc_neighbors(True)
        layer_sels.append(layer_sel)

    ss_sels = []  # alpha, beta, coil
    for ss in ["H", "E", "L"]:
        ss_sel = SecondaryStructureSelector()
        ss_sel.set_selected_ss(ss)
        ss_sel.set_overlap(0)
        ss_sel.set_minH(3)
        ss_sel.set_minE(2)
        ss_sel.set_use_dssp(True)
        if ss == "L":
            ss_sel.set_include_terminal_loops(True)
        ss_sels.append(ss_sel)

    helix_cap_sel = AndResidueSelector(
        ss_sels[2], PrimarySequenceNeighborhoodSelector(1, 0, ss_sels[0], False)
    )
    helix_start_sel = AndResidueSelector(
        ss_sels[0], PrimarySequenceNeighborhoodSelector(0, 1, helix_cap_sel, False)
    )
    final_ss_sels = [
        helix_start_sel,
        AndResidueSelector(ss_sels[0], NotResidueSelector(helix_start_sel)),
        ss_sels[1],
        AndResidueSelector(ss_sels[2], NotResidueSelector(helix_cap_sel)),
    ]

    region_sels = [helix_cap_sel]
    for layer_sel, ss_sel in product(layer_sels, final_ss_sels):
        region_sels.append(AndResidueSelector(layer_sel, ss_sel))

    layer_design = {}
    for layer_name, layer_aas, region_sel in zip(
        layer_names, layer_aas_list, region_sels
    ):
        layer_design[layer_name] = (layer_aas, region_sel)

    return layer_design


def gen_task_factory(
    design_sel: ResidueSelector = None,
    pack_sel: ResidueSelector = None,
    pack_nbhd: bool = False,
    pack_nondesignable: bool = False,
    extra_rotamers_level: int = 0,
    include_current: bool = False,
    limit_arochi: bool = False,
    prune_buns: bool = False,
    upweight_ppi: bool = False,
    restrict_pro_gly: bool = False,
    restrict_dslf: bool = False,
    include_native: bool = True,
    precompute_ig: bool = False,
    ifcl: bool = False,
    layer_design: dict = None,
) -> TaskFactory:
    """
    :param: design_sel: ResidueSelector for designable residues.
    :param: pack_nbhd: bool, whether to pack the neighborhood of designable residues.
    :param: extra_rotamers_level: int, how many extra rotamers to add to the packer.
    :param: limit_arochi: bool, whether to limit extreme aromatic chi angles.
    :param: prune_buns: bool, whether to prune rotamers with buried unsatisfied polars.
    :param: upweight_ppi: bool, whether to upweight interfaces.
    :param: restrict_pro_gly: bool, whether to restrict proline and glycine from design.
    :param: precompute_ig: bool, whether to precompute the interaction graph.
    :param: ifcl: bool, whether to initialize the packer from the command line.
    :param: layer_design: dict, custom layer design definition if you want to use one.
    :return: TaskFactory for design.
    Sets up the TaskFactory for design.
    """

    import pyrosetta
    from pyrosetta.rosetta.core.pack.task.operation import (
        OperateOnResidueSubset,
        PreventRepackingRLT,
        RestrictAbsentCanonicalAASRLT,
        RestrictAbsentCanonicalAASExceptNativeRLT,
        RestrictToRepackingRLT,
    )
    from pyrosetta.rosetta.core.select.residue_selector import(
        AndResidueSelector,
        FalseResidueSelector,
        NeighborhoodResidueSelector,
        OrResidueSelector,
        ResidueNameSelector,
    )

    task_factory = pyrosetta.rosetta.core.pack.task.TaskFactory()

    if not design_sel:
        design_sel = FalseResidueSelector()

    if not pack_sel:
        pack_sel = FalseResidueSelector()

    if pack_nbhd:
        # pack around designable area
        pack_nbhd_sel = NeighborhoodResidueSelector(design_sel, 6, True)
        # everything not designable
        pack_op = OperateOnResidueSubset(RestrictToRepackingRLT(), design_sel, True)
        task_factory.push_back(pack_op)

        # everything neither designable nor packable
        design_nbhd_or_pack_sel = OrResidueSelector(pack_nbhd_sel, pack_sel)
        lock_op = OperateOnResidueSubset(
            PreventRepackingRLT(), design_nbhd_or_pack_sel, True
        )
        task_factory.push_back(lock_op)
    elif pack_nondesignable:
        # everything not designable
        pack_op = OperateOnResidueSubset(RestrictToRepackingRLT(), design_sel, True)
        task_factory.push_back(pack_op)
    else:
        # everything not designable
        pack_op = OperateOnResidueSubset(RestrictToRepackingRLT(), design_sel, True)
        task_factory.push_back(pack_op)
        # everything neither designable nor packable
        design_or_pack_sel = OrResidueSelector(design_sel, pack_sel)
        lock_op = OperateOnResidueSubset(
            PreventRepackingRLT(), design_or_pack_sel, True
        )
        task_factory.push_back(lock_op)

    if extra_rotamers_level > 0:
        extra_rotamers_op = (
            pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
        )
        extra_rotamers_op.ex1(True)
        extra_rotamers_op.ex2(False)
        extra_rotamers_op.ex3(False)
        extra_rotamers_op.ex4(False)
        if extra_rotamers_level > 1:
            extra_rotamers_op.ex2(True)
            if extra_rotamers_level > 2:
                extra_rotamers_op.ex3(True)
                if extra_rotamers_level > 3:
                    extra_rotamers_op.ex4(True)
        task_factory.push_back(extra_rotamers_op)

    if include_current:
        native_op = (
            pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent()
        )
        task_factory.push_back(native_op)

    if prune_buns:
        prune_buns_op = (
            pyrosetta.rosetta.protocols.task_operations.PruneBuriedUnsatsOperation()
        )
        prune_buns_op.allow_even_trades(False)
        prune_buns_op.atomic_depth_cutoff(3.5)
        prune_buns_op.minimum_hbond_energy(-1.0)
        task_factory.push_back(prune_buns_op)

    # add standard task operations
    if limit_arochi:
        arochi_op = (
            pyrosetta.rosetta.protocols.task_operations.LimitAromaChi2Operation()
        )
        arochi_op.chi2max(110)
        arochi_op.chi2min(70)
        arochi_op.include_trp(True)
        task_factory.push_back(arochi_op)

    if upweight_ppi:
        upweight_ppi_op = (
            pyrosetta.rosetta.protocols.pack_interface.ProteinProteinInterfaceUpweighter()
        )
        upweight_ppi_op.set_weight(3.0)
        task_factory.push_back(upweight_ppi_op)

    if restrict_pro_gly:
        pro_gly_sel = ResidueNameSelector("PRO,GLY")
        pro_gly_op = OperateOnResidueSubset(PreventRepackingRLT(), pro_gly_sel, False)
        task_factory.push_back(pro_gly_op)
    
    if restrict_dslf:
        restrict_dslf_op = (
            pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides()
        )
        task_factory.push_back(restrict_dslf_op)

    if precompute_ig:
        ig_op = pyrosetta.rosetta.protocols.task_operations.SetIGTypeOperation(
            False, False, False, True
        )
        task_factory.push_back(ig_op)

    if ifcl:
        ifcl_op = pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline()
        task_factory.push_back(ifcl_op)

    if layer_design:
        for layer_aas, region_sel in layer_design.values():
            if include_native:
                task_op = RestrictAbsentCanonicalAASExceptNativeRLT()
            else:
                task_op = RestrictAbsentCanonicalAASRLT()
            task_op.aas_to_keep(layer_aas)
            # if you disallow the native of a residue that is only allowed to pack and that native is
            # not in the list of allowed resis for that layer, then it won't pack at all
            design_region_sel = AndResidueSelector(design_sel, region_sel)
            region_op = OperateOnResidueSubset(task_op, design_region_sel, False)
            task_factory.push_back(region_op)

    return task_factory


def gen_scorefxn(
    cartesian: bool = False,
    res_type_constraint: bool = False,
    hbonds: bool = False,
    weights: str = "beta_nov16.wts",
) -> ScoreFunction:
    """
    TODO
    Requires the following init flags:
    `-corrections::beta_nov16 true` or `-corrections::beta_nov16_cart true`
    :param: cartesian: score function set up for Cartesian scoring
    :param: res_type_constraint: score function set up for designing with residue type information, as in FavorNativeResidue
    :param: hbonds: score function set up for designing hbonds
    """

    import pyrosetta

    sfxn = pyrosetta.create_score_function(weights)

    if cartesian:
        sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5)
        sfxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0)

    if res_type_constraint:
        sfxn.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
        )

    if hbonds:
        # a scorefunction that likes hbonds- I got this from Ryan Kibler, not sure of the original source. Bcov?
        # use with prune_buns (allow_even_trades=False, atomic_depth_cutoff=3.5, minimum_hbond_energy=-1.0)
        hbond_options = pyrosetta.rosetta.core.scoring.hbonds.HBondOptions()
        hbond_options.use_hb_env_dep(True)
        energy_method_options = (
            pyrosetta.rosetta.core.scoring.methods.EnergyMethodOptions()
        )
        energy_method_options.hbond_options(hbond_options)
        energy_method_options.approximate_buried_unsat_penalty_burial_atomic_depth(3.5)
        energy_method_options.approximate_buried_unsat_penalty_hbond_bonus_cross_chain(
            -7.0
        )
        energy_method_options.approximate_buried_unsat_penalty_hbond_bonus_ser_to_helix_bb(
            1.0
        )
        energy_method_options.approximate_buried_unsat_penalty_hbond_energy_threshold(
            -1.0
        )
        energy_method_options.approximate_buried_unsat_penalty_natural_corrections1(
            True
        )
        sfxn.set_energy_method_options(energy_method_options)
        sfxn.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreType.approximate_buried_unsat_penalty,
            17.0,
        )

    return sfxn


# False is the default for all these in a new movemap
def gen_movemap(
    jump: bool = False,
    chi: bool = False,
    bb: bool = False,
    nu: bool = False,
    branch: bool = False,
) -> MoveMap:
    """
    :param: jump: bool, whether to allow jump moves.
    :param: chi: bool, whether to allow chi moves.
    :param: bb: bool, whether to allow backbone moves.
    :param: nu: bool, whether to allow nu moves (for carbohydrates).
    :param: branch: bool, whether to allow branch moves (for carbohydrates).
    :return: MoveMap for design.
    Sets up the MoveMap for design.
    """

    import pyrosetta

    movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
    movemap.set_jump(jump)
    movemap.set_chi(chi)
    movemap.set_bb(bb)
    movemap.set_nu(nu)
    movemap.set_branches(branch)
    return movemap


def fast_design(
    pose: Pose,
    task_factory: TaskFactory,
    scorefxn: ScoreFunction,
    movemap: MoveMap,
    relax_script: str = "InterfaceDesign2019",
    repeats: int = 1,
    cartesian: bool = False,
) -> None:
    """
    :param: pose: Pose, the pose to design.
    :param: task_factory: TaskFactory, the task factory to use.
    :param: scorefxn: ScoreFunction, the score function to use.
    :param: movemap: MoveMap, the movemap to use.
    :param: repeats: int, the number of times to repeat the design.
    :return: None.
    Runs FastDesign with the given task factory and score function.
    """

    import pyrosetta

    # use an xml to create the fastdesign mover since it's easier to load a relax script
    # and to specify the minimization algorithm
    # chose lbfgs_armigo_nonmonotone for the minimization algorithm based on
    # https://new.rosettacommons.org/docs/wiki/rosetta_basics/structural_concepts/minimization-overview
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <FastDesign name="fastdesign" repeats="{repeats}" ramp_down_constraints="false" 
                batch="false" cartesian="{str(cartesian).lower()}" bondangle="false" bondlength="false" 
                min_type="lbfgs_armijo_nonmonotone" relaxscript="{relax_script}">
            </FastDesign>
        </MOVERS>
        """
    )
    fdes_mover = objs.get_mover("fastdesign")
    fdes_mover.set_task_factory(task_factory)
    fdes_mover.set_scorefxn(scorefxn)
    fdes_mover.set_movemap(movemap)
    fdes_mover.apply(pose)
    return


def fast_relax(
    pose: Pose,
    task_factory: TaskFactory,
    scorefxn: ScoreFunction,
    movemap: MoveMap,
    relax_script: str = "MonomerRelax2019",
    repeats: int = 5,
    cartesian: bool = False,
    coord_constrain_sidechains: bool = False,
    constrain_relax_to_start_coords: bool = False,
) -> None:
    """
    :param: pose: Pose, the pose to design.
    :param: task_factory: TaskFactory, the task factory to use.
    :param: scorefxn: ScoreFunction, the score function to use.
    :param: movemap: MoveMap, the movemap to use.
    :param: relax_script: str, the relax script to use.
    :param: repeats: int, the number of times to repeat the design.
    :param: cartesian: bool, whether to use cartesian minimization.
    :param: coord_constrain_sidechains: bool, whether to constrain sidechains to their
    starting coordinates. (Need sfxn that ends with `_cst` to use this, and needs to be
    in cartesian mode I think.)
    :param: constrain_relax_to_start_coords: bool, whether to constrain the relax to
    starting coordinates. (Need sfxn that ends with `_cst` to use this, and needs to be
    in cartesian mode I think.)
    :return: None.
    Runs FastRelax with the given task factory and score function.
    """

    import pyrosetta

    # use an xml to create the fastdesign mover since it's easier to load a relax script
    # and to specify the minimization algorithm
    # chose lbfgs_armigo_nonmonotone for the minimization algorithm based on
    # https://new.rosettacommons.org/docs/wiki/rosetta_basics/structural_concepts/minimization-overview
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <FastRelax name="fastrelax" repeats="{repeats}" ramp_down_constraints="false" 
                batch="false" cartesian="{str(cartesian).lower()}" bondangle="false" bondlength="false" 
                min_type="lbfgs_armijo_nonmonotone" relaxscript="{relax_script}">
            </FastRelax>
        </MOVERS>
        """
    )
    frelax_mover = objs.get_mover("fastrelax")
    frelax_mover.set_task_factory(task_factory)
    frelax_mover.set_scorefxn(scorefxn)
    frelax_mover.set_movemap(movemap)
    if coord_constrain_sidechains:
        frelax_mover.coord_constrain_sidechains(True)
    else:
        pass
    if constrain_relax_to_start_coords:
        frelax_mover.constrain_relax_to_start_coords(True)
    else:
        pass
    frelax_mover.apply(pose)
    return


def pack_rotamers(
    pose: Pose, task_factory: TaskFactory, scorefxn: ScoreFunction
) -> None:
    """
    :param: pose: Pose, the pose to design.
    :param: task_factory: TaskFactory, the task factory to use.
    :param: scorefxn: ScoreFunction, the score function to use.
    :return: None.
    Runs PackRotamers with the given task factory and score function.
    """
    import pyrosetta

    pack_mover = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
    pack_mover.task_factory(task_factory)
    pack_mover.score_function(scorefxn)
    pack_mover.apply(pose)


def struct_profile(pose: Pose, design_sel: ResidueSelector) -> None:
    """
    :param: pose: Pose, the pose to design.
    :param: design_sel: ResidueSelector, the design selection.
    :return: None.
    Runs StructProfile with the given design selection.
    """

    import pyrosetta

    # used the defaults from rosettascripts, only changing things changed in the original one-state xml
    # need to specify all these options since there are limited constructors available for structprofilemover
    sp_mover = pyrosetta.rosetta.protocols.simple_moves.StructProfileMover(
        rmsThreshold=0.6,
        burialThreshold=3.0,
        consider_topN_frags=100,
        burialWt=0.0,
        only_loops=False,
        censorByBurial=False,
        allowed_deviation=0.1,
        allowed_deviation_loops=0.1,
        eliminate_background=True,
        psiblast_style_pssm=False,
        outputProfile=False,
        add_csts_to_pose=True,
        ignore_terminal_res=True,
        fragment_store_path="",
        fragment_store_format="hashed",
        fragment_store_compression="all",
    )
    sp_mover.set_residue_selector(design_sel)
    sp_mover.apply(pose)


def seq_profile(
    pose: Pose,
    design_sel: Optional[ResidueSelector] = None,
    matrix: Optional[str] = "IDENTITY",
    weight: Optional[float] = 1.0,
    scaling: Optional[str] = "prob",
    seq_profile_source: Optional[str] = "current",
    seq_profile_fname: Optional[str] = "",
) -> None:
    """
    :param: pose: Pose, the pose to design.
    :param: design_sel: ResidueSelector, the selection of residues to which to apply the sequence profile.
    :param: matrix: str, allowed values: "IDENTITY", "MATCH", "BLOSUM62"
    :param: weight: float, adjust the post-scaling strength of the constraints.
    :param: scaling: str, allowed values: "prob", "global", "none". Set how to scale the given values. "prob" does a Boltzmann-weighted probability based on the profile score (the unweighted scores for all 20 amino acid identities at any given position sum to -1.0). "global" does a global linear fixed-zero rescaling such that all (pre-weighted) values fall in the range of -1.0 to 1.0. "none" does no adjustment of values.
    :param: seq_profile_source: str, allowed values: "current", "starting", "native", "fasta", "pssm", "pdb". The source of the sequence profile to use. Current uses the current structure, starting uses the starting input structure, native uses the structure specified by -in:file:native, fasta uses the sequence specified by -in:file:fasta, pssm uses the sequence profile in a blast-formatted pssm file specified by seq_profile_fname, pdb uses the pdb file specified by seq_profile_fname.
    :param: seq_profile_source: str, the path to a file containing the sequence profile.
    :return: None.
    Runs StructProfile with the given design selection. Example usage: to penalize changing the existing residues in a certain region
    of the pose selected by design_sel, use the following: seq_profile(pose, design_sel, "IDENTITY", 1.0, "prob", "current")
    """

    import pyrosetta

    if seq_profile_source == "current":
        src_str = 'use_current="true"'
    # starting may behave strangely outside of rosettascripts, not sure...
    elif seq_profile_source == "starting":
        src_str = 'use_starting="true"'
    # need to specify -in:file:native
    elif seq_profile_source == "native":
        src_str = 'use_native="true"'
    # need to specify -in:file:fasta
    elif seq_profile_source == "fasta":
        src_str = 'use_fasta="true"'
    # need to point seq_profile_fname to the pssm file
    elif seq_profile_source == "pssm":
        src_str = f'pssm="{seq_profile_fname}"'
    # need to point seq_profile_fname to the pssm file
    elif seq_profile_source == "pdb":
        src_str = f'pdbname="{seq_profile_fname}"'
    else:
        raise ValueError(
            f"seq_profile_source must be one of 'current', 'starting', 'native', 'fasta', 'pssm', 'pdb', not {seq_profile_source}"
        )

    if design_sel is not None:
        nondesignable_sel = (
            pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(
                design_sel
            )
        )
        exclude_resnums_str = (
            ' exclude_resnums="'
            + ",".join(
                str(i)
                for i in pyrosetta.rosetta.core.select.get_residues_from_subset(
                    nondesignable_sel.apply(pose)
                )
            )
            + '"'
        )
    # can only set the string_exclude_resnums_ internal variable by parsing an xml tag... annoying
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <MOVERS>
            <FavorSequenceProfile name="seq_profile" matrix={matrix} weight="{weight}" scaling="{scaling}" {src_str}{exclude_resnums_str}/>
        </MOVERS>
        """
    )
    seq_profile_mover = objs.get_mover("seq_profile")
    seq_profile_mover.apply(pose)


def clear_constraints(pose: Pose) -> None:
    """
    :param: pose: Pose, the pose to design.
    :return: None.
    Removes all constraints from the pose.
    """
    import pyrosetta

    cst_mover = pyrosetta.rosetta.protocols.constraint_movers.ClearConstraintsMover()
    cst_mover.apply(pose)


def gen_sasa_filter(pose: Pose, name: str = "sasa"):
    pass
    # skip this for now, since it's annoying that it can only compute across jumps
    # import pyrosetta
    # sasa_filter = pyrosetta.rosetta.protocols.simple_filters.InterfaceSasaFilter()


def gen_score_filter(scorefxn: ScoreFunction, name: str = "score") -> Filter:
    """
    :param: scorefxn: ScoreFunction, the score function to use.
    :param: name: str, the name of the filter.
    :return: ScoreFilter, the score filter.
    Generates a score filter with the given score function.
    """
    import pyrosetta

    score_filter = pyrosetta.rosetta.protocols.score_filters.ScoreTypeFilter(
        scorefxn, pyrosetta.rosetta.core.scoring.ScoreType.total_score, 0
    )
    score_filter.set_user_defined_name(name)
    return score_filter


def score_on_chain_subset(
    pose: Pose, filter: Filter, chain_list: list
) -> Union[float, int, str]:
    """
    :param: pose: Pose, the pose to score.
    :param: filter: Filter, the filter to use.
    :param: chain_list: list, the list of chains to score.
    :return: Union[float, int, str], the score of the pose with the filter.
    Scores the pose with the given filter on the given chains.
    """

    import pyrosetta

    chain_str = "".join(str(i) for i in chain_list)
    name = filter.get_user_defined_name() + "_" + chain_str
    sw_mover = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
    sw_mover.chain_order(chain_str)
    mb_filter = pyrosetta.rosetta.protocols.filters.MoveBeforeFilter(
        mover=sw_mover, filter=filter
    )
    mb_filter.set_user_defined_name(name)
    # maybe could use score instead of report_sm, but couldn't figure out how to set scorename_ of the filter so the values are written to the pdb...
    value = mb_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(value))
    return value


def score_cms(
    pose: Pose,
    sel_1: ResidueSelector,
    sel_2: ResidueSelector,
    name: str = "cms",
    filtered_area: float = 250.0,
    distance_weight: float = 1.0,
    quick: bool = False,
    use_rosetta_radii: bool = False,  # default values for the filter
) -> float:
    """
    :param: pose: Pose, the pose to score.
    :param: sel_1: ResidueSelector, the first selection.
    :param: sel_2: ResidueSelector, the second selection.
    :param: name: str, the name of the filter.
    :param: filtered_area: float, the area of the interface to consider.
    :param: distance_weight: float, the weight to apply to the distance.
    :param: quick: bool, whether to use the quick version of the filter.
    :param: use_rosetta_radii: bool, whether to use the rosetta radii.
    :return: float, the score of the pose with the filter.
    Scores the contact molecular surface area between the two selections.
    Requires pyrosetta >= 2021.44
    """

    import pyrosetta

    cms_filter = (
        pyrosetta.rosetta.protocols.simple_filters.ContactMolecularSurfaceFilter(
            filtered_area=filtered_area,
            distance_weight=distance_weight,
            quick=quick,
            verbose=False,  # constructor requires these arguments
            selector1=sel_1,
            selector2=sel_2,
            use_rosetta_radii=use_rosetta_radii,
        )
    )
    cms_filter.set_user_defined_name(name)
    cms = cms_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(cms))
    return cms


def score_ddg(
    pose: Pose,
    jump: int = 1,
    repeats: int = 1,
    repack: bool = True,
    repack_bound: bool = True,
    repack_unbound: bool = True,
    relax_bound: bool = False,
    relax_unbound: bool = True,
    name: str = "ddg"
) -> float:
    """
    :param: pose: Pose, the pose to score.
    :param: jump: the jump to calculate DDG across
    :param: name: str, the name of the filter.
    :return: float, the score of the pose with the filter.
    Scores the pose ddg at a particular jump.
    """
    import pyrosetta

    relax_mover = ""
    if relax_bound or relax_unbound:
        relax_mover = \
            f"""
            <SCOREFXNS>
                <ScoreFunction name="sfxn" weights="beta_nov16" />
            </SCOREFXNS>
            <TASKOPERATIONS>
                <ProteinInterfaceDesign name="pack_long" design_chain1="0" design_chain2="0" jump="{jump}" interface_distance_cutoff="15"/>
            </TASKOPERATIONS>
            <MOVERS>
                <TaskAwareMinMover name="min" scorefxn="sfxn" bb="1" chi="1" task_operations="pack_long" />
            </MOVERS>
            """

    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""{relax_mover}
        <FILTERS>
            <Ddg name="ddg" jump="{jump}" repeats="{repeats}" repack="{int(repack)}" repack_bound="{int(repack_bound)}" repack_unbound="{int(repack_unbound)}" relax_bound="{int(relax_bound)}" relax_unbound="{int(relax_unbound)}" />
        </FILTERS>
        """
    )
    ddg_filter = objs.get_filter("ddg")
    ddg_filter.set_user_defined_name(name)
    ddg = ddg_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(ddg))
    return ddg


def score_SAP(
    pose: Pose,
    name: str = "SAP",
    sap_calculate_selector: ResidueSelector = None,
    sasa_selector: ResidueSelector = None,
    score_selector: ResidueSelector = None,
) -> float:
    """
    :param: pose: Pose, the pose to score.
    :param: name: str, the name of the filter.
    :return: float, the score of the pose with the filter.
    Scores the pose with the worst9mer filter on the given selector. Should not be used
    over selections that include jumps/chainbreaks.
    """
    import pyrosetta

    sap_metric = pyrosetta.rosetta.core.pack.guidance_scoreterms.sap.SapScoreMetric()
    if sap_calculate_selector is not None:
        sap_metric.set_calculate_selector(sap_calculate_selector)
    if sasa_selector is not None:
        sap_metric.set_sasa_selector(sasa_selector)
    if score_selector is not None:
        sap_metric.set_score_selector(score_selector)
    sap = sap_metric.calculate(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(sap))
    return sap


def score_sc(
    pose: Pose, sel_1: ResidueSelector, sel_2: ResidueSelector, name: str = "sc_int"
) -> float:
    """
    :param: pose: Pose, the pose to score.
    :param: sel_1: ResidueSelector, the first selection.
    :param: sel_2: ResidueSelector, the second selection.
    :param: name: str, the name of the score.
    :return: float, the shape complementary score of the pose.
    Scores the the shape complementarity between the two selections on a pose.
    """

    import pyrosetta

    sc_filter = pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter()
    sc_filter.selector1(sel_1)
    sc_filter.selector2(sel_2)
    sc_filter.set_user_defined_name(name)
    sc_filter.write_int_area(True)
    sc_filter.write_median_distance(True)
    sc = sc_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(sc))
    return sc


def score_ss_sc(
    pose: Pose, helices: bool = True, loops: bool = True, name: str = "ss_sc"
) -> float:
    """
    :param: pose: Pose, the pose to score.
    :param: helices: bool, whether to score helices.
    :param: loops: bool, whether to score loops.
    :param: name: str, the name of the score.
    :return: float, the score of the pose with the score.
    Scores the pose with the secondary structure shape complementarity filter.
    """
    import pyrosetta

    ss_sc_filter = (
        pyrosetta.rosetta.protocols.denovo_design.filters.SSShapeComplementarityFilter()
    )
    ss_sc_filter.set_calc_helices(helices)
    ss_sc_filter.set_calc_loops(loops)
    ss_sc_filter.set_user_defined_name(name)
    ss_sc = ss_sc_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(ss_sc))
    return ss_sc


def score_wnm(pose: Pose, sel: ResidueSelector = None, name: str = "wnm") -> float:
    """
    :param: pose: Pose, the pose to score.
    :param: sel: ResidueSelector, the selector to use.
    :param: name: str, the name of the filter.
    :return: float, the score of the pose with the filter.
    Scores the pose with the worst9mer filter on the given selector. Should not be used
    over selections that include jumps/chainbreaks.
    """
    import pyrosetta

    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        """
        <FILTERS>
            <Worst9mer name="wnm" rmsd_lookup_threshold="0.4" confidence="0" />
        </FILTERS>
        """
    )
    wnm_filter = objs.get_filter("wnm").subfilter()
    wnm_filter.set_user_defined_name(name)
    if sel is not None:
        # this seems to be broken, I get "AttributeError: 'pyrosetta.rosetta.protocols.filters.StochasticFilt' object has no attribute 'set_residue_selector'"
        wnm_filter.set_residue_selector(sel)
    wnm = wnm_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(wnm))
    return wnm


def score_wnm_all(pose: Pose, name: str = "wnm") -> List[float]:
    """
    :param: pose: Pose, the pose to score.
    :param: name: str, the name of the filter.
    :return: List[float], the list of scores for the pose.
    Scores all the chains of a pose seperately on worst9mer.
    """
    # loading the database takes 4.5 minutes, but once loaded, remains for the rest of the python session
    # could instead call score_wnm inside this function, but that would require parsing the xml multiple times. Faster to just parse it once and change the residue selector.
    import pyrosetta

    # using an xml to create the worst9mer filter because I couldn't figure out how to use pyrosetta without completely crashing python
    # this is probably because there are many internal variables in the filter which only get set when parsing an xml
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        """
        <FILTERS>
            <Worst9mer name="wnm" rmsd_lookup_threshold="0.4" confidence="0" />
        </FILTERS>
        """
    )
    wnm_filter = objs.get_filter("wnm")
    wnm_filter.set_user_defined_name(name)
    # use this method since having trouble with the set_residue_selector method
    wnms = []
    for chain_num in range(1, pose.num_chains() + 1):
        wnm = score_on_chain_subset(pose, wnm_filter, [chain_num])
        wnms.append(wnm)
    return wnms


def score_wnm_helix(pose: Pose, name: str = "wnm_hlx") -> float:
    """
    :param: pose: Pose to score.
    :param: name: Name of the score.
    :return: The helical worst9mer score.
    Score the helical worst9mer of the pose.
    """
    import pyrosetta

    # using an xml to create the worst9mer filter because I couldn't figure out how to use pyrosetta without completely crashing python
    # this is probably because there are many internal variables in the filter which only get set when parsing an xml
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        """
        <FILTERS>
            <Worst9mer name="wnm_hlx" rmsd_lookup_threshold="0.4" confidence="0" only_helices="true" />
        </FILTERS>
        """
    )
    wnm_hlx_filter = objs.get_filter("wnm_hlx")
    wnm_hlx_filter.set_user_defined_name(name)
    wnm = wnm_hlx_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(wnm))
    return wnm


def score_per_res(
    pose: Pose, scorefxn: ScoreFunction, name: str = "score"
) -> Tuple[float, float]:
    """
    :param: pose: Pose to score.
    :param: scorefxn: ScoreFunction to use.
    :param: name: Name of the score.
    :return: The score of the pose.
    Calculates the score per res of the pose using the scorefxn.
    """
    import pyrosetta

    score_filter = gen_score_filter(scorefxn, name)
    score = score_filter.report_sm(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(score))
    score_pr = score / pose.size()
    pyrosetta.rosetta.core.pose.setPoseExtraScore(
        pose, name + "_per_res", str(score_pr)
    )
    return score, score_pr


def score_CA_dist(pose: Pose, resi_1: int, resi_2: int, name: str = "dist") -> float:
    """
    :param: pose: Pose to measure CA distance.
    :param: resi_1: Residue index of first residue.
    :param: resi_2: Residue index of second residue.
    :param: name: Name of score.
    :return: Distance between residues.
    Measures the CA distance between two residues.
    """

    import sys
    from pathlib import Path

    import pyrosetta

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.states import measure_CA_dist

    dist = measure_CA_dist(pose, resi_1, resi_2)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, name, str(dist))
    return dist


def score_loop_dist(pose: Pose, pre_break_helix: int, name: str = "loop_dist") -> float:
    """
    :param: pose: Pose to measure loop distance.
    :param: pre_break_helix: Helix index of the helix before the loop.
    :param: name: Name of score.
    :return: Distance between helices.
    Measures the distance between two helices.
    """

    import sys
    from pathlib import Path

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.states import StateMaker

    ends = StateMaker.get_helix_endpoints(pose, n_terminal=False)
    end = (
        ends[pre_break_helix] + 1
    )  # now plus 1 since the helix ends one residue earlier due to the new chainbreak
    loop_dist = score_CA_dist(pose, end, end + 1, name)
    return loop_dist

def score_sasa(pose: Pose, hydrophobic: bool = False, polar : bool = False, residues : str = None, name : str = "sasa") -> float:
    """
    :param: pose: Pose, the pose to score.
    :param: hydrophobic: Only score SASA for hydrophobic atoms?
    :param: polar: Only score SASA for polar atoms?
    :param: residues: Comma-separated list of residues over which to calculate SASA.
    :param: name: str, the name of the filter.
    :return: float, the score of the pose with the filter.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pack.task.operation import (
        OperateOnResidueSubset,
        PreventRepackingRLT,
    )
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
    assert (not hydrophobic or not polar), "Hydrophobic and polar cannot both be set to True.  For all-atom SASA, set both to False."
    hydrophobic=str(hydrophobic).lower()
    polar=str(polar).lower()
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        f"""
        <FILTERS>
            <TotalSasa name="sasa" hydrophobic="{hydrophobic}" polar="{polar}" />
        </FILTERS>
        """
    )
    sasa_filter = objs.get_filter("sasa")
    if residues is not None:
        task_factory=pyrosetta.rosetta.core.pack.task.TaskFactory()
        task_factory.push_back(OperateOnResidueSubset(PreventRepackingRLT(), ResidueIndexSelector(residues), True))
        sasa_filter.task_factory(task_factory)
    sasa_filter.set_user_defined_name(name)
    return sasa_filter.report_sm(pose)


@requires_init
def one_state_design_bound_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a packed pose to use as a starting point for interface
    design. If None, a pose will be generated from the input pdb_path.
    :param: kwargs: keyword arguments for the design.
    :return: an iterator of packed poses.
    Requires the following init flags:
    `-corrections::beta_nov16 true`
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()
    # hardcode precompute_ig
    not_fixed_sel = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector()
    int_sel = interface_among_chains(chain_list=[1, 2, 3], vector_mode=True)
    design_sel = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(
        int_sel, not_fixed_sel
    )
    print_timestamp("Generated interface selector", start_time=start_time)
    layer_design = gen_std_layer_design()
    task_factory = gen_task_factory(
        design_sel=design_sel,
        pack_nbhd=False,
        extra_rotamers_level=2,
        limit_arochi=True,
        prune_buns=False,
        upweight_ppi=True,
        restrict_pro_gly=True,
        precompute_ig=False,
        ifcl=True,
        layer_design=layer_design,
    )
    print_timestamp("Generated interface design task factory", start_time=start_time)

    clean_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn = pyrosetta.create_score_function("beta_nov16.wts")
    design_sfxn.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.res_type_constraint, 1.0
    )
    print_timestamp("Generated score functions", start_time=start_time)

    fixbb_mm = gen_movemap(jump=True, chi=True, bb=False)
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

    for pose in poses:
        start_time = time()

        scores = dict(pose.scores)

        # don't design any fixed residues
        fixed_sel = (
            pyrosetta.rosetta.core.select.residue_selector.FalseResidueSelector()
        )
        if "fixed_resis" in scores:
            fixed_resi_str = scores["fixed_resis"]
            # handle an empty string
            if fixed_resi_str:
                fixed_sel = (
                    pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
                        fixed_resi_str
                    )
                )
        not_fixed_sel.set_residue_selector(fixed_sel)

        # for the neighborhood residue selector
        pose.update_residue_neighbors()

        print_timestamp(
            "Starting 1 round of fixed backbone design...",
            end="",
            start_time=start_time,
        )
        fast_design(
            pose=pose,
            task_factory=task_factory,
            scorefxn=design_sfxn,
            movemap=fixbb_mm,
            repeats=1,
        )
        print("complete.")
        print_timestamp(
            "Starting 2 rounds of flexible backbone design...",
            end="",
            start_time=start_time,
        )
        fixbb_pose = pose.clone()
        # run fast design twice, flexible backbone movemap and yield the pose both times
        for _ in range(2):
            pose = fixbb_pose.clone()
            fast_design(
                pose=pose,
                task_factory=task_factory,
                scorefxn=design_sfxn,
                movemap=flexbb_mm,
                repeats=2,
            )
            print("complete.")

            print_timestamp("Scoring loop distance...", end="", start_time=start_time)
            pre_break_helix = int(float(scores["pre_break_helix"]))
            score_loop_dist(pose, pre_break_helix, name="loop_dist")
            print("complete.")

            print_timestamp(
                "Scoring contact molecular surface and shape complementarity...",
                end="",
                start_time=start_time,
            )
            an_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(1)
            ac_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(2)
            b_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(3)
            dhr_sel = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(
                an_sel, ac_sel
            )
            selector_pairs = [
                (an_sel, ac_sel),
                (an_sel, b_sel),
                (ac_sel, b_sel),
                (b_sel, dhr_sel),
            ]
            pair_names = ["AnAc", "AnB", "AcB", "AnAcB"]
            for (sel_1, sel_2), name in zip(selector_pairs, pair_names):
                score_cms(pose, sel_1, sel_2, "cms_" + name)
                score_sc(pose, sel_1, sel_2, "sc_" + name)
            print("complete.")

            print_timestamp(
                "Scoring secondary structure shape complementarity...",
                end="",
                start_time=start_time,
            )
            score_ss_sc(pose)
            print("complete.")

            print_timestamp("Scoring...", end="", start_time=start_time)
            score_per_res(pose, clean_sfxn)
            score_filter = gen_score_filter(clean_sfxn)
            add_metadata_to_pose(pose, "path_in", pdb_path)
            print("complete.")
            end_time = time()
            total_time = end_time - start_time
            print_timestamp(
                f"Total time: {total_time:.2f} seconds", start_time=start_time
            )
            add_metadata_to_pose(pose, "time", total_time)
            clear_terms_from_scores(pose)

            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def score_binder(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a packed pose to use as a starting point for interface
    design. If None, a pose will be generated from the input pdb_path.
    :param: kwargs: keyword arguments for the design.
    :return: an iterator of packed poses.
    Requires the following init flags:
    `-corrections::beta_nov16 true`
    """

    import sys
    from pathlib import Path

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    cluster_scores = kwargs.pop("df_scores", True)
    move_target = kwargs.pop("move_target", False)

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=cluster_scores, pack_result=False
        )

    chA_sel = ChainSelector("A")
    chB_sel = ChainSelector("B")

    sfxn = gen_scorefxn()
    task_factory = gen_task_factory(
        pack_nondesignable = True,
        extra_rotamers_level = 2,
        include_current = True,
        limit_arochi = True,
        upweight_ppi = True,
        precompute_ig = True
    )
    movemap = gen_movemap(jump=True, chi=True, bb=True)

    for pose in poses:
        if not move_target:
            movemap.set_bb(False)
            movemap.set_bb_true_range(1, pose.chain_end(1))

        fast_relax(pose, task_factory, sfxn, movemap, "InterfaceDesign2019")

        score_cms(pose, chA_sel, chB_sel)
        score_ddg(pose, jump=1, repeats=3, repack=True, repack_bound=True, repack_unbound=True, relax_bound=True, relax_unbound=True)
        score_sc(pose, chA_sel, chB_sel)
        score_ss_sc(pose)
        score_per_res(pose, sfxn)

        ppose = io.to_packed(pose)
        yield ppose
