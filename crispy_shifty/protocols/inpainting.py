# Python standard library
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# 3rd party library imports

# Rosetta library imports
from pyrosetta.distributed import requires_init
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose

class InpaintingRunner(ABC):
    """
    Class for running ProteinInpainting on any cluster.
    """

    def __init__(
        self,
        pose: Pose,
        input_file: Optional[str] = None,
        contigs: Optional[str] = None,
        inpaint_seq: Optional[str] = None,
        inpaint_str: Optional[str] = None,
        res_translate: Optional[str] = None,
        tie_translate: Optional[str] = None,
        block_rotate: Optional[str] = None,
        multi_templates: Optional[str] = None,
        num_designs: Optional[int] = 100,
        topo_pdb: Optional[str] = None,
        topo_conf: Optional[str] = None,
        topo_contigs: Optional[str] = None,
        temperature: Optional[float] = None,
        min_decoding_distance: Optional[int] = None,
        save_original_pose: Optional[bool] = False,
        **kwargs,
    ):
        """
        """

        import os
        from pathlib import Path

        import git

        self.pose = pose
        self.input_file = input_file
        self.contigs = contigs
        self.inpaint_seq = inpaint_seq
        self.inpaint_str = inpaint_str
        self.res_translate = res_translate
        self.tie_translate = tie_translate
        self.block_rotate = block_rotate
        self.multi_templates = multi_templates
        self.num_designs = num_designs
        self.topo_pdb = topo_pdb
        self.topo_conf = topo_conf
        self.topo_contigs = topo_contigs
        self.temperature = temperature
        self.min_decoding_distance = min_decoding_distance
        self.save_original_pose = save_original_pose

        self.scores = dict(self.pose.scores)

        # set up standard flags for inpainting
        all_flags = {
            "--contigs": self.contigs,
            "--inpaint_seq": self.inpaint_seq,
            "--inpaint_str": self.inpaint_str,
            "--res_translate": self.res_translate,
            "--tie_translate": self.tie_translate,
            "--block_rotate": self.block_rotate,
            "--multi_templates": self.multi_templates,
            "--num_designs": self.num_designs,
            "--topo_pdb": self.topo_pdb,
            "--topo_conf": self.topo_conf,
            "--topo_contigs": self.topo_contigs,
            "--temperature": self.temperature,
            "--min_decoding_distance": self.min_decoding_distance,
        }
        self.flags = {k: v for k, v in all_flags.items() if v is not None}

        self.allowed_flags = [
            "--pdb",
            "--out",
            "--contigs",
            "--inpaint_seq",
            "--inpaint_str",
            "--res_translate",
            "--tie_translate",
            "--block_rotate",
            "--multi_templates",
            "--num_designs",
            "--topo_pdb",
            "--topo_conf",
            "--topo_contigs",
            "--temperature",
            "--min_decoding_distance"
        ]
        
        # use git to find the root of the repo
        repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        self.python = str(Path(root) / "envs" / "shifty" / "bin" / "python")
        if os.path.exists(self.python):
            pass
        else:  # shifty env must be installed in envs/shifty or must be used on DIGS
            self.python = "/projects/crispy_shifty/envs/shifty/bin/python"
        self.script = str(
            Path(__file__).parent.parent.parent / "proteininpainting" / "inpaint.py"
        )
        self.tmpdir = None  # this will be updated by the setup_tmpdir method.
        self.command = None  # this will be updated by the setup_runner method.
        self.is_setup = False  # this will be updated by the setup_runner method.

    def get_command(self) -> str:
        """
        :return: command to run.
        """
        return self.command

    def get_flags(self) -> Dict[str, str]:
        """
        :return: dictionary of flags.
        """
        return self.flags

    def get_script(self) -> str:
        """
        :return: script path.
        """
        return self.script

    def get_tmpdir(self) -> str:
        """
        :return: temporary directory path.
        """
        return self.tmpdir

    def set_script(self, script: str) -> None:
        """
        :param: script: The path to the script.
        :return: None.
        """
        self.script = script
        self.update_command()
        return

    def setup_tmpdir(self) -> None:
        """
        :return: None
        Create a temporary directory for the InpaintingRunner. Checks for various best
        practice locations for the tmpdir in the following order: TMPDIR, PSCRATCH,
        CSCRATCH, /net/scratch. Uses the cwd if none of these are available.
        """
        import os
        import pwd
        import uuid

        if os.environ.get("TMPDIR") is not None:
            tmpdir_root = os.environ.get("TMPDIR")
        elif os.environ.get("PSCRATCH") is not None:
            tmpdir_root = os.environ.get("PSCRATCH")
        elif os.environ.get("CSCRATCH") is not None:
            tmpdir_root = os.environ.get("CSCRATCH")
        elif os.path.exists("/net/scratch"):
            tmpdir_root = f"/net/scratch/{pwd.getpwuid(os.getuid()).pw_name}"
        else:
            tmpdir_root = os.getcwd()

        self.tmpdir = os.path.join(tmpdir_root, uuid.uuid4().hex)
        os.makedirs(self.tmpdir, exist_ok=True)
        return

    def teardown_tmpdir(self) -> None:
        """
        :return: None
        Remove the temporary directory for the InpaintingRunner.
        """
        import shutil

        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)
        return

    def update_command(self) -> None:
        """
        :return: None
        Update the command to run.
        """
        self.command = " ".join(
            [
                f"{self.python} {self.script}",
                f"--pdb {self.input_file}",
                " ".join([f"{k} {v}" for k, v in self.flags.items()]),
            ]
        )

    def update_flags(self, update_dict: Dict[str, str]) -> None:
        """
        :param: update_dict: dictionary of flags to update.
        :return: None
        Update the flags dictionary with the provided dictionary.
        Validate the flags before updating.
        """

        for flag in update_dict.keys():
            if flag not in self.allowed_flags:
                raise ValueError(
                    f"Flag {flag} is not allowed. Allowed flags are {self.allowed_flags}"
                )
        self.flags.update(update_dict)
        return

    # @abstractmethod
    def configure_structure_flags(self, input_pose: Pose) -> Pose:
        """
        A child class of InpaintingRunner can reimplement this function with additional functionality.
        This should set the contigs string and perhaps the inpaint_seq string for the specific application.
        This can also be used to modify the input structure before inpainting, for example by adding guiding fragments algorithmically.
        """
        self.contigs = ",0 ".join(self.contigs.split(":"))
        self.update_flags({"--contigs": self.contigs})
        return input_pose

    def setup_runner(
        self, file: Optional[str] = None, flag_update: Optional[Dict[str, str]] = None
    ) -> None:
        """
        :param: file: path to input file. If None, use the dumped tmp.pdb.
        :param: flag_update: dictionary of flags to update, if any.
        :return: None
        Setup the InpaintingRunner.
        Create a temporary directory for the InpaintingRunner.
        Dump the pose temporarily to a PDB file in the temporary directory.
        Update the flags dictionary with the provided dictionary if any.
        Setup the command line arguments for the InpaintingRunner.
        """
        import os
        import pyrosetta
        import pyrosetta.distributed.io as io
        from pyrosetta.rosetta.core.pose import setPoseExtraScore, clearPoseExtraScores

        # setup the tmpdir
        self.setup_tmpdir()
        out_path = self.tmpdir
        # set input_file
        if file is not None:
            self.input_file = file
        else:
            self.input_file = os.path.join(out_path, "tmp.pdb")
        # write the pose to a clean PDB file of only ATOM coordinates.
        tmp_pdb_path = os.path.join(out_path, "tmp.pdb")
        new_pose = self.pose.clone()
        clearPoseExtraScores(new_pose) # highly important, otherwise pdbstrings in the scores get added to the pose lol
        new_pose = self.configure_structure_flags(new_pose)
        pdbstring = io.to_pdbstring(new_pose)
        with open(tmp_pdb_path, "w") as f:
            f.write(pdbstring)
        if self.save_original_pose:
            setPoseExtraScore(self.pose, "original_pose", pdbstring)
        # update the flags with the path to the tmpdir
        self.update_flags({"--out": out_path + "/inpaint"})
        if flag_update is not None:
            self.update_flags(flag_update)
        else:
            pass
        self.update_command()
        print("Command to run: ", self.command)
        self.is_setup = True
        return

    def generate_inpaints(self) -> Iterator[PackedPose]:
        """
        :param: pose: Pose object in which to inpaint a fusion.
        :return: None
        Run inpainting on the provided pose in a subprocess.
        Read the results from the temporary directory and store them in the pose.
        Remove the temporary directory.
        """
        import os
        import sys
        from glob import glob
        import numpy as np
        from pathlib import Path

        import pyrosetta
        import pyrosetta.distributed.io as io
        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.utils.io import cmd

        assert self.is_setup, "InpaintingRunner is not set up."

        # run the command in a subprocess
        out_err = cmd(self.command)
        print(out_err)
        # or could the following print in real time?
        # print(cmd(self.command))

        inpaints = []
        trb_files = glob(os.path.join(self.tmpdir, "inpaint*.trb"))
        for trb_file in trb_files:
            raw_scores = np.load(trb_file, allow_pickle=True)
            # process scores
            inpaint_scores = {
                "inpaint_mean_lddt": np.mean(raw_scores["inpaint_lddt"]),
                "contigs": ';'.join(raw_scores["contigs"]),
                "sampled_mask": ';'.join(raw_scores["sampled_mask"]),
                "inpaint_seq_resis": ','.join(str(i) for i, mask in enumerate(raw_scores["inpaint_seq"], start=1) if not mask),
                "inpaint_str_resis": ','.join(str(i) for i, mask in enumerate(raw_scores["inpaint_str"], start=1) if not mask),
                "inpaint_length": len(raw_scores["inpaint_lddt"]),
                "total_length": len(raw_scores["inpaint_seq"]),
            }
    
            inpaint_pose = io.to_pose(io.pose_from_file(trb_file[:-3]+"pdb"))
            for k, v in self.scores.items():
                setPoseExtraScore(inpaint_pose, k, v)
            for k, v in inpaint_scores.items():
                setPoseExtraScore(inpaint_pose, k, v)
            
            inpaints.append(io.to_packed(inpaint_pose))

        self.teardown_tmpdir()

        for inpaint in inpaints:
            yield inpaint

class InpaintingStructure(InpaintingRunner):
    """
    Class for inpainting additional structure into an existing pose, guided by roughly positioned secondary structure elements.
    TODO add fixed residue selection argument
    """

    def __init__(
        self,
        rough_contigs: str,
        rough_tie_translate: Optional[str] = None,
        contig_len_style: Optional[str] = "input", # possible values are "input" and "ss_length"
        fixed_resis: Optional[str] = None,
        additional_inpaint_seq: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        :param: args: arguments to pass to InpaintingRunner.
        :param: kwargs: keyword arguments to pass to InpaintingRunner.
        Initialize the base class for inpainting runners with common attributes.
        """
        super().__init__(*args, **kwargs)
        self.rough_contigs = rough_contigs
        self.rough_tie_translate = rough_tie_translate
        self.contig_len_style = contig_len_style
        self.fixed_resis = fixed_resis
        self.additional_inpaint_seq = additional_inpaint_seq

    def configure_structure_flags(self, input_pose: Pose) -> Pose:
        """
        Configure contigs and inpaint_seq flags for building structure. 
        Configure the inpaint_seq flag by an interfacebyvector selector between the 
        modeled helices and the other chains being inpainted to.
        Rough contig entry format: "MinPad-MaxPad,ChainNumber_TrimToLen,MinPad-MaxPad"
        The rough contig string must start and end with padding entries, unless there 
        is only one entry (an intact chain as context).
        Chains in the input pose must be ordered in the order they appear in the
        rough_contigs string.
        """

        import sys
        from more_itertools import interleave_longest
        from pathlib import Path
        import pyrosetta

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.design import interface_among_chains

        def pose_chain_to_entry(begin, end, pdbinfo):
            chain = pdbinfo.chain(begin)
            begin = pdbinfo.number(begin)
            end = pdbinfo.number(end)
            return f"{chain}{begin}-{end}"

        pose = input_pose.clone()
        pdbinfo = pose.pdb_info()

        contigs = []
        inpaint_seq = []
        for rough_contig in self.rough_contigs.split(':'):
            rough_contig_list = rough_contig.split(',')

            if len(rough_contig_list) == 1:
                chain_num = int(rough_contig_list[0])
                contigs.append(pose_chain_to_entry(pose.chain_begin(chain_num), pose.chain_end(chain_num), pdbinfo))
                continue

            # get the termini and trimming of each chain
            between_data_list = []
            chain_data_list = []
            for i, chain_contig in enumerate(rough_contig_list):
                if i % 2:
                    chain_contig = chain_contig.split('_')
                    chain_num = int(chain_contig[0])
                    trim_len = int(chain_contig[1])
                    begin = pose.chain_begin(chain_num)
                    end = pose.chain_end(chain_num)
                    begin_fragment = begin
                    end_fragment = end
                    if trim_len:
                        begin_fragment = (end - begin) // 2 + begin - trim_len // 2 + 1
                        end_fragment = begin_fragment + trim_len - 1
                    chain_data_list.append((chain_num, trim_len, begin_fragment, end_fragment, begin, end))
                else:
                    chain_contig = chain_contig.split('-')
                    pad_min = int(chain_contig[0])
                    pad_max = int(chain_contig[1])
                    between_data_list.append((pad_min, pad_max))

            # get the interfaces to redesign during inpainting
            int_sel = interface_among_chains([c[0] for c in chain_data_list], True)
            # this list will be kept in pose numbering instead of pdb numbering
            inpaint_seq_resis = set(pyrosetta.rosetta.core.select.get_residues_from_subset(int_sel.apply(pose)))

            # generate the portions of chains to keep during inpainting
            pdbinfo = pose.pdb_info()
            chain_contigs = []
            for chain_data in chain_data_list:
                chain_num = chain_data[0]
                begin = chain_data[2]
                end = chain_data[3]
                chain_contigs.append(pose_chain_to_entry(begin, end, pdbinfo))
                if chain_data[1]:
                    # don't inpaint seq of resis that are being trimmed
                    inpaint_seq_resis.difference_update(range(chain_data[4], chain_data[2]-1))
                    inpaint_seq_resis.difference_update(range(chain_data[3]+1, chain_data[5]))
                    # do inpaint seq of all resis in the guiding chunk
                    inpaint_seq_resis.update(range(chain_data[2], chain_data[3]+1))
            if self.additional_inpaint_seq:
                inpaint_seq_resis.update(int(r) for r in self.additional_inpaint_seq.split(','))
            if self.fixed_resis:
                inpaint_seq_resis.difference_update(int(r) for r in self.fixed_resis.split(','))
            
            # generate the lengths to inpaint between chains
            between_chain_contigs = []
            for chain_1_data, between_data, chain_2_data in zip(chain_data_list[:-1], between_data_list[1:-1], chain_data_list[1:]):
                end_1 = chain_1_data[3]
                begin_2 = chain_2_data[2]
                trim_1 = chain_1_data[1]
                trim_2 = chain_2_data[1]
                pad_min = between_data[0]
                pad_max = between_data[1]

                # use the length of the original elements to decide on the contig length plus padding for the loop
                if self.contig_len_style == "input":
                    inpaint_length = begin_2 - end_1 - 1
                    between_chain_contigs.append(f"{inpaint_length+pad_min}-{inpaint_length+pad_max}")
                # calculate the length of secondary structure required to connect two points
                elif self.contig_len_style == "ss_length":
                    # if between two nontrimming chains, calculate loop length by the distance between the termini divided by the length per residue of a PP2 conformation (3A)
                    if (not trim_1) and (not trim_2):
                        direct_dist = pose.residue(end_1).xyz("CA").distance(pose.residue(begin_2).xyz("CA"))
                        # pp2 loop length, minus 1 since both attachment residues are already present
                        inpaint_length = int(round(direct_dist / 3)) - 1
                        between_chain_contigs.append(f"{inpaint_length+pad_min}-{inpaint_length+pad_max}")
                    # if between two chains to trim, use the length of the original elements to decide on the contig length plus padding for the loop
                    elif trim_1 and trim_2:
                        inpaint_length = begin_2 - end_1 - 1
                        between_chain_contigs.append(f"{inpaint_length+pad_min}-{inpaint_length+pad_max}")
                    # if between a trimming chain and a nontrimming chain, calculate loop length by the distance between the termini divided by the length per residue of a helix (1.5A), plus 2-4 for the loop
                    else:
                        direct_dist = pose.residue(end_1).xyz("CA").distance(pose.residue(begin_2).xyz("CA"))
                        # helix connection length, minus 1 since both attachment residues are already present, plus 4 because this tends to underestimate by about one turn (helices really go parallel instead of directly between termini)
                        inpaint_length = int(round(direct_dist / 1.5)) + 3
                        between_chain_contigs.append(f"{inpaint_length+pad_min}-{inpaint_length+pad_max}")
                else:
                    raise ValueError(f"contig_len_style {self.contig_len_style} not recognized")
            
            # check for lengths to inpaint before and after the first and last chains
            before_contig = ""
            inpaint_length = 0
            pad_min = between_data_list[0][0]
            pad_max = between_data_list[0][1]
            if chain_data_list[0][1]: # if the first chain is trimmed
                begin_fragment = chain_data_list[0][2]
                begin = chain_data_list[0][4]
                inpaint_length = begin_fragment - begin
            if inpaint_length + pad_max > 0:
                before_contig = f"{inpaint_length+pad_min}-{inpaint_length+pad_max},"
            after_contig = ""
            inpaint_length = 0
            pad_min = between_data_list[-1][0]
            pad_max = between_data_list[-1][1]
            if chain_data_list[-1][1]: # if the last chain is trimmed
                end_fragment = chain_data_list[-1][3]
                end = chain_data_list[-1][5]
                inpaint_length = end - end_fragment
            if inpaint_length + pad_max > 0:
                after_contig = f",{inpaint_length+pad_min}-{inpaint_length+pad_max}"
            
            contigs.append(before_contig + ','.join(interleave_longest(chain_contigs, between_chain_contigs)) + after_contig)
            inpaint_seq += sorted(list(inpaint_seq_resis))
        
        self.contigs = ',0 '.join(contigs)
        self.inpaint_seq = ','.join(f"{pdbinfo.chain(resi)}{pdbinfo.number(resi)}" for resi in sorted(inpaint_seq))
        
        print("inpaint_seq_resis:", '+'.join([r[1:] for r in self.inpaint_seq.split(',')]))

        update_flags_dict = {
            "--contigs": self.contigs,
            "--inpaint_seq": self.inpaint_seq,
        }

        if self.rough_tie_translate:
            tie_translates = []
            for rough_tie_trans in self.rough_tie_translate.split(':'):
                rough_tie_trans_list = rough_tie_trans.split(',')
                tie_trans_list = []
                for chain_num in rough_tie_trans_list[:-1]:
                    chain_num = int(chain_num)
                    tie_trans_list.append(pose_chain_to_entry(pose.chain_begin(chain_num), pose.chain_end(chain_num), pdbinfo))
                tie_trans_list.append(rough_tie_trans_list[-1])
                tie_translates.append(','.join(tie_trans_list))
            self.tie_translate = ':'.join(tie_translates)
            update_flags_dict["--tie_translate"] = self.tie_translate

        self.update_flags(update_flags_dict)

        return pose


@requires_init
def inpaint(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:


    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.inpainting import InpaintingRunner
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs.pop("pdb_path")
        cluster_scores = kwargs.pop("df_scores", True)
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=cluster_scores, pack_result=False
        )

    for pose in poses:
        print_timestamp("Setting up inpainting runner", start_time)
        # construct the InpaintingRunner object
        runner = InpaintingRunner(
            pose=pose,
            **kwargs
        )
        runner.setup_runner()
        print_timestamp("Running inpainting", start_time)
        # generate inpaints
        for ppose in runner.generate_inpaints():
            yield ppose


@requires_init
def inpaint_structure(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:


    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.inpainting import InpaintingStructure
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs.pop("pdb_path")
        cluster_scores = kwargs.pop("df_scores", True)
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=cluster_scores, pack_result=False
        )

    rough_contigs = kwargs.pop("rough_contigs", None)
    rough_tie_translate = kwargs.pop("rough_tie_translate", None)

    for pose in poses:
        print_timestamp("Setting up inpainting runner", start_time)
        # construct the InpaintingStructure object
        runner = InpaintingStructure(
            pose=pose,
            rough_contigs=rough_contigs,
            rough_tie_translate=rough_tie_translate,
            **kwargs
        )
        runner.setup_runner()
        print_timestamp("Running inpainting", start_time)
        # generate inpaints
        for ppose in runner.generate_inpaints():
            yield ppose
