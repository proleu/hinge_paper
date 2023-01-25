# Python standard library
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

from pyrosetta.distributed import requires_init

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector


########## Azobenzene Crosslinking ##########
# The main function for azobenzene crosslinking.  Use with the following init:
# pyrosetta.init(
#    " ".join(
#        [
#            "-beta",
#            "-corrections:gen_potential true",
#            "-out:level 100",
#            "-in:file:extra_res_fa AZC.params",
#            "-in:file:extra_res_fa AZT.params",
#            "-gen_bonded_params_file scoring/score_functions/generic_potential/generic_bonded.aamerge.txt",
#            "-genbonded_score_hybrid true",
#        ]
#    )
# )
# params files can be found in crispy_shifty/params
# Minimal usage: add_azo(pose, selection, residue_name)
# pose - input pose
# selection - ResidueIndexSelector or str specifying two residues
# residue_name - What is the crosslinker called?
# Other options:
# add_constraints - bool, should constraints be added to the crosslinker?
# filter_by_sidechain_distance - float, negative for no filtering, positive to set a threshold in angstroms for crosslinking to be attempted.  Use this in a try/except block as it will throw an AssertionError
# filter_by_cst_energy - float, negative for no filtering, positive to set a threshold in Rosetta energy units (for constraint scores only) for crosslinking to be accepted.  Use in a try/except block
# filter_by_total_score - same as filter_by_cst_energy, but total_score instead of constraint scores
# force_cys - bool, True to mutate selected residues to Cys, False to error if one of the selected residues is not Cys
# sc_fast_relax_rounds - int, number of round of fast relax for the linked sidechains and linker only
# final_fast_relax_rounds - int, number of round of whole-structure fast relax to do after relaxing sidechains and linker
# custom_movemap - override the default movemap for the final relax (untested)
# rmsd_filter - None to not calculate rmsd.  {"sele":StrOfIndices, "super_sele":StrOfIndices, "type":str(pyrosetta.rosetta.core.scoring.rmsd_atoms member), "save":bool, "cutoff":float}
### Disulfide crosslinking ###

@requires_init
def generate_hinge_staples(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:

    # state_chains is the chain numbers of state X and state Y separated by a comma
    # bounds is the end of the N-terminal side of the hinge and the start of the C-terminal side of the hinge separated by a comma
    # does not support 

    import numpy as np
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.conformation import form_disulfide
    from pyrosetta.rosetta.core.pose import setPoseExtraScore
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
    from stapler import NativeDisulfideStapler
    from crispy_shifty.protocols.alignment import score_rmsd
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import fast_relax, gen_movemap, gen_scorefxn, gen_task_factory, score_ss_sc, score_wnm, score_wnm_helix

    # generate poses or convert input packed pose into pose
    pdb_path = kwargs.pop("pdb_path")
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        # skip the kwargs check
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    state_chains = [int(i) for i in kwargs["state_chains"].split(",")]
    bounds = None
    if "bounds" in kwargs:
        bounds = [int(i) for i in kwargs["bounds"].split(',')]
    compute_post_relax_scores = False
    if "compute_post_relax_scores" in kwargs and kwargs["compute_post_relax_scores"].lower() == "true":
        compute_post_relax_scores = True
    yield_alt_staples = False
    if "yield_alt_staples" in kwargs and kwargs["yield_alt_staples"].lower() == "true":
        yield_alt_staples = True
    hash_table_fname = None
    if "hash_table_fname" in kwargs:
        hash_table_fname = kwargs["hash_table_fname"]

    for pose in poses:
        scores = dict(pose.scores)

        if bounds is None:
            if "dslf_bounds" in scores:
                bounds = [int(i) for i in scores["dslf_bounds"].split(',')]
            elif "new_loop_str" in scores:
                new_loop_resis = scores["new_loop_str"].split(",")
                bounds = [int(new_loop_resis[0]), int(new_loop_resis[-1])]
            else:
                center = (pose.chain_end(state_chains[0]) - pose.chain_begin(state_chains[0]) - 1) // 2
                bounds = [center, center+1]

        hinge_parts = [ResidueIndexSelector(),ResidueIndexSelector()]
        hinge_parts[0].set_index_range(1, bounds[0])
        hinge_parts[1].set_index_range(bounds[1], pose.chain_end(state_chains[0]) - pose.chain_begin(state_chains[0]) - 1)

        if hash_table_fname:
            stapler = NativeDisulfideStapler(
                residue_selectors=hinge_parts,
                minimum_sequence_distance=4,
                hash_table_kwargs={'key_type' : np.dtype('i8'), 'value_type': np.dtype('i4'), 'filename': hash_table_fname}
            )
        else:
            stapler = NativeDisulfideStapler(
                residue_selectors=hinge_parts,
                minimum_sequence_distance=4,
            )

        state_poses = []
        state_stapled_poses = []
        for chain_num in state_chains:
            state_pose = pose.split_by_chain(chain_num)
            state_poses.append(state_pose)
            state_stapled_poses.append(list(stapler.apply(state_pose.clone())))

        # for each state, put the staples into the alternate state(s) 
        # evaluate whether they are likely to form in that alternate state, and discard if so
        # OR
        # record the parameters of the staples in the alternate state(s) as scores so you can filter later.
        # parameters for filtering: rosetta disulfide energy, Cb-Cb distance, dot product of Ca-Cb vectors, something else?

        movemap = gen_movemap(True, True, True)
        beta_cart_sfxn = gen_scorefxn(cartesian=True)
        task_factory = gen_task_factory(pack_nondesignable=True, extra_rotamers_level=2, limit_arochi=True, precompute_ig=True)

        for i, stapled_poses in enumerate(state_stapled_poses):
            for stapled_pose in stapled_poses:
                for k, v in scores.items():
                    setPoseExtraScore(stapled_pose, k, v)
                setPoseExtraScore(stapled_pose, "dslf_state", str(i))

                cyd_1_index = stapled_pose.sequence().index("C") + 1 # for 1-indexing
                cyd_2_index = stapled_pose.sequence().index("C", cyd_1_index) + 1
                setPoseExtraScore(stapled_pose, "dslf_resis", f"{cyd_1_index}_{cyd_2_index}")

                cyd_1_resi = stapled_pose.residue(cyd_1_index)
                cyd_2_resi = stapled_pose.residue(cyd_2_index)
                Cb_dist = cyd_1_resi.xyz("CB").distance(cyd_2_resi.xyz("CB"))
                Ca_Cb_1 = (cyd_1_resi.xyz("CA") - cyd_1_resi.xyz("CB")).normalize()
                Ca_Cb_2 = (cyd_2_resi.xyz("CA") - cyd_2_resi.xyz("CB")).normalize()
                Ca_Cb_dot = Ca_Cb_1.dot_product(Ca_Cb_2)
                setPoseExtraScore(stapled_pose, "dslf_Cb_dist", Cb_dist)
                setPoseExtraScore(stapled_pose, "dslf_Ca_Cb_dot", Ca_Cb_dot)

                # generate metrics to evaluate whether the staples are likely to form in the other states
                for j, alt_state_pose in enumerate(state_poses):
                    # skip comparing to the same state
                    if i == j:
                        continue
                    alt_stapled_pose = alt_state_pose.clone()

                    # this mutates the two residues to cysteines and tells Rosetta they are connected by a disulfide bond,
                    # but does not repack and minimize to satisfy the bond geometry
                    form_disulfide(alt_stapled_pose.conformation(), cyd_1_index, cyd_2_index)
                    cyd_1_resi = alt_stapled_pose.residue(cyd_1_index)
                    cyd_2_resi = alt_stapled_pose.residue(cyd_2_index)
                    Cb_dist = cyd_1_resi.xyz("CB").distance(cyd_2_resi.xyz("CB"))
                    Ca_Cb_1 = (cyd_1_resi.xyz("CA") - cyd_1_resi.xyz("CB")).normalize()
                    Ca_Cb_2 = (cyd_2_resi.xyz("CA") - cyd_2_resi.xyz("CB")).normalize()
                    Ca_Cb_dot = Ca_Cb_1.dot_product(Ca_Cb_2)
                    setPoseExtraScore(stapled_pose, f"dslf_state_{j}_Cb_dist", Cb_dist)
                    setPoseExtraScore(stapled_pose, f"dslf_state_{j}_Ca_Cb_dot", Ca_Cb_dot)

                    # if the disulfide is poor in the alternate state, the disulfide bond constraint causes the backbone to be relaxed into a weird position to 
                    # satisfy the disulfide. Try to capture this weirdness to evaluate how good the disulfide is in the alternate state. Perhaps a combination 
                    # of worst9mer, ss_sc, and/or RMSD between pre- and post-relaxed conformations
                    if compute_post_relax_scores:
                        relaxed_alt_stapled_pose = alt_stapled_pose.clone()
                        fast_relax(relaxed_alt_stapled_pose, task_factory, beta_cart_sfxn, movemap, "MonomerRelax2019", 5, True)
                        delta_ss_sc = score_ss_sc(relaxed_alt_stapled_pose) - score_ss_sc(alt_stapled_pose)
                        delta_wnm = score_wnm(relaxed_alt_stapled_pose) - score_wnm(alt_stapled_pose)
                        delta_wnm_hlx = score_wnm_helix(relaxed_alt_stapled_pose) - score_wnm_helix(alt_stapled_pose)
                        rmsd = score_rmsd(stapled_pose, relaxed_alt_stapled_pose, name=f"dslf_state_{j}_rlx_rmsd")
                        setPoseExtraScore(stapled_pose, f"dslf_state_{j}_delta_ss_sc", delta_ss_sc)
                        setPoseExtraScore(stapled_pose, f"dslf_state_{j}_delta_wnm", delta_wnm)
                        setPoseExtraScore(stapled_pose, f"dslf_state_{j}_delta_wnm_hlx", delta_wnm_hlx)

                    if yield_alt_staples:
                        setPoseExtraScore(alt_stapled_pose, "dslf_resis", f"{cyd_1_index}_{cyd_2_index}")
                        setPoseExtraScore(alt_stapled_pose, "dslf_state", f"{i}_alt_{j}")
                        setPoseExtraScore(alt_stapled_pose, "dslf_Cb_dist", Cb_dist)
                        setPoseExtraScore(alt_stapled_pose, "dslf_Ca_Cb_dot", Ca_Cb_dot)
                        if compute_post_relax_scores:
                            setPoseExtraScore(alt_stapled_pose, "dslf_rlx_rmsd", rmsd)
                            setPoseExtraScore(alt_stapled_pose, "dslf_delta_ss_sc", delta_ss_sc)
                            setPoseExtraScore(alt_stapled_pose, "dslf_delta_wnm", delta_wnm)
                            setPoseExtraScore(alt_stapled_pose, "dslf_delta_wnm_hlx", delta_wnm_hlx)
                        yield io.to_packed(alt_stapled_pose)

                yield io.to_packed(stapled_pose)

def get_helix_endpoints(pose: Pose) -> dict:
    """
    Use dssp to get the endpoints of helices.
    Make a dictionary of the start (n_terminal=True) or end residue indices of
    helices in the pose.
    :param pose: Pose to get endpoints from.
    :param n_terminal: If True, get the start residue indices of helices.
    :return: Dictionary of start or end residue indices of helices in the pose.
    """
    import pyrosetta

    ss = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    # first make a dictionary of all helical residues, indexed by helix number
    helix_list = []
    for i in range(1, len(pose.sequence())):
        if (ss.get_dssp_secstruct(i) == "H") & (
            ss.get_dssp_secstruct(i - 1) != "H"
        ):
            helix_list.append([i])
        if (ss.get_dssp_secstruct(i) == "H") & (
            ss.get_dssp_secstruct(i + 1) != "H"
        ):
            helix_list[-1].append(i)
    
    return helix_list

@requires_init
def disulfide_spammer(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:

    # place all possible disulfides between all combinations of secondary structural elements

    import numpy as np
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import setPoseExtraScore
    from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
    from stapler import NativeDisulfideStapler
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
    else:
        # skip the kwargs check
        pdb_path = kwargs.pop("pdb_path")
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    hash_table_fname = None
    if "hash_table_fname" in kwargs:
        hash_table_fname = kwargs["hash_table_fname"]

    for pose in poses:
        scores = dict(pose.scores)

        # get the helix endpoints
        # could also use the SSElementSelector for this
        helix_endpoints = get_helix_endpoints(pose)

        for helix_ind, helix_ends in enumerate(helix_endpoints[:-1]):

            construct_parts = [ResidueIndexSelector(), ResidueIndexSelector()]
            for prev_ends in helix_endpoints[helix_ind+1:]:
                for r in range(prev_ends[0], prev_ends[1] + 1):
                    construct_parts[0].append_index(r)
            construct_parts[1].set_index_range(helix_ends[0], helix_ends[-1])

            if hash_table_fname:
                stapler = NativeDisulfideStapler(
                    residue_selectors=construct_parts,
                    minimum_sequence_distance=4,
                    hash_table_kwargs={'key_type' : np.dtype('i8'), 'value_type': np.dtype('i4'), 'filename': hash_table_fname}
                )
            else:
                stapler = NativeDisulfideStapler(
                    residue_selectors=construct_parts,
                    minimum_sequence_distance=4,
                )

            for i, stapled_pose in enumerate(stapler.apply(pose.clone())):
                for k, v in scores.items():
                    setPoseExtraScore(stapled_pose, k, v)
                setPoseExtraScore(stapled_pose, "dslf_state", f"{helix_ind}_{i}")

                cyd_1_index = stapled_pose.sequence().index("C") + 1 # for 1-indexing
                cyd_2_index = stapled_pose.sequence().index("C", cyd_1_index) + 1
                setPoseExtraScore(stapled_pose, "dslf_resis", f"{cyd_1_index}_{cyd_2_index}")

                cyd_1_resi = stapled_pose.residue(cyd_1_index)
                cyd_2_resi = stapled_pose.residue(cyd_2_index)
                Cb_dist = cyd_1_resi.xyz("CB").distance(cyd_2_resi.xyz("CB"))
                Ca_Cb_1 = (cyd_1_resi.xyz("CA") - cyd_1_resi.xyz("CB")).normalize()
                Ca_Cb_2 = (cyd_2_resi.xyz("CA") - cyd_2_resi.xyz("CB")).normalize()
                Ca_Cb_dot = Ca_Cb_1.dot_product(Ca_Cb_2)
                setPoseExtraScore(stapled_pose, "dslf_Cb_dist", Cb_dist)
                setPoseExtraScore(stapled_pose, "dslf_Ca_Cb_dot", Ca_Cb_dot)

                yield io.to_packed(stapled_pose)
