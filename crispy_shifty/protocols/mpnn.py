# Python standard library
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

# 3rd party library imports
import numpy as np
from pyrosetta.distributed import requires_init

# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector

# Custom library imports


def dict_to_fasta(
    seqs_dict: Dict[str, str],
    out_path: str,
) -> None:
    """
    :param seqs_dict: dictionary of sequences to write to a fasta file.
    :param out_path: path to write the fasta file to.
    :return: None
    Write a fasta file to the provided path with the provided sequence dict.
    """
    import os
    from pathlib import Path

    # make the output path if it doesn't exist
    if not os.path.exists(Path(out_path).parent):
        os.makedirs(Path(out_path).parent)
    else:
        pass
    # write the sequences to a fasta file
    with open(out_path, "w") as f:
        for i, seq in seqs_dict.items():
            f.write(f">{i}\n{seq}\n")
    return


def fasta_to_dict(fasta: str, new_tags: bool = False) -> Dict[str, str]:
    """
    :param fasta: fasta filepath to read from.
    :param new_tags: if False, use the sequence tag as the key, if True, use the index.
    :return: dictionary of tags and sequences.
    Read in a fasta file and return a dictionary of tags and sequences.
    """
    seqs_dict = {}

    with open(fasta, "r") as f:
        i = 0
        for line in f:
            if line.startswith(">"):  # this is a header line
                if new_tags:
                    tag = str(i)
                else:
                    tag = line.strip().replace(">", "")
                seqs_dict[tag] = ""
                i += 1
            else:  # we can assume this is a sequence line, add the sequence
                seqs_dict[tag] += line.strip()

    return seqs_dict


def levenshtein_distance(seq_a: str, seq_b: str) -> int:
    """
    :param seq_a: first sequence to compare.
    :param seq_b: second sequence to compare.
    :return: levenshtein distance between the two sequences.
    Calculate the levenshtein distance between two sequences.
    """
    # https://en.wikipedia.org/wiki/Levenshtein_distance
    # initialize distance matrix
    distance_matrix = np.zeros((len(seq_a) + 1, len(seq_b) + 1))
    for id1 in range(len(seq_a) + 1):
        distance_matrix[id1][0] = id1
    for id2 in range(len(seq_b) + 1):
        distance_matrix[0][id2] = id2
    a = 0
    b = 0
    c = 0
    for id1 in range(1, len(seq_a) + 1):
        for id2 in range(1, len(seq_b) + 1):
            if seq_a[id1 - 1] == seq_b[id2 - 1]:
                distance_matrix[id1][id2] = distance_matrix[id1 - 1][id2 - 1]
            else:
                a = distance_matrix[id1][id2 - 1]
                b = distance_matrix[id1 - 1][id2]
                c = distance_matrix[id1 - 1][id2 - 1]
                if a <= b and a <= c:
                    distance_matrix[id1][id2] = a + 1
                elif b <= a and b <= c:
                    distance_matrix[id1][id2] = b + 1
                else:
                    distance_matrix[id1][id2] = c + 1
    levenshtein_distance = int(distance_matrix[id1][id2])
    return levenshtein_distance


@requires_init
def thread_full_sequence(
    pose: Pose,
    sequence: str,
    start_res: int = 1,
) -> Pose:
    """
    :param: pose: Pose to thread sequence onto.
    :param: sequence: Sequence to thread onto pose.
    :return: Pose with threaded sequence.
    Threads a full sequence onto a pose after cloning the input pose.
    Doesn't require the chainbreak '/' to be cleaned.
    """
    from pyrosetta.rosetta.protocols.simple_moves import SimpleThreadingMover

    try:
        assert sequence.count("/") == 0
    except AssertionError:
        print("Cleaning chain breaks in sequence.")
        sequence = sequence.replace("/", "")

    pose = pose.clone()
    stm = SimpleThreadingMover()
    stm.set_sequence(sequence, start_res)
    stm.apply(pose)

    return pose


class MPNNRunner(ABC):
    """
    Abstract base class for MPNN runners.
    """

    import os
    import pwd
    import shutil
    import subprocess
    import uuid

    import pyrosetta.distributed.io as io

    def __init__(
        self,
        batch_size: Optional[int] = 8,
        model_name: Optional[str] = "v_48_010",
        path_to_model_weights: Optional[str] = "/databases/mpnn/vanilla_model_weights/",
        num_sequences: Optional[int] = 64,
        omit_AAs: Optional[str] = "X",
        temperature: Optional[float] = 0.1,
        design_selector: Optional[ResidueSelector] = None,
        chains_to_mask: Optional[List[str]] = None,
        mpnn_subtype: Optional[str] = "vanilla",
        **kwargs,
    ):
        """
        Initialize the base class for MPNN runners with common attributes.
        :param: batch_size: number of sequences to generate per batch.
        :param: model_name: name of the model to use. v_48_010 is probably best, v_32*
        variants use less memory.
        :param: path_to_model_weights: absolute path to model weights. Change if running
        somewhere besides the digs.
        :param: num_sequences: number of sequences to generate in total.
        :param: omit_AAs: concatenated string of amino acids to omit from the sequence.
        :param: temperature: temperature to use for the MPNN.
        :param: design_selector: ResidueSelector that specifies residues to design.
        :param: chains_to_mask: list of chains to mask, these will be designable.
        If no `chains_to_mask` is provided, the runner will run on (mask) all chains.
        If a `chains_to_mask` is provided, the runner will run on (mask) only that chain.
        If no `design_selector` is provided, all residues on all masked chains will be designed.
        The chain letters in your PDB must be correct.
        Does not allow the use of --score_only, --conditional_probs_only, or
        --conditional_probs_only_backbone.
        """

        from pathlib import Path

        self.batch_size = batch_size
        self.model_name = model_name
        self.path_to_model_weights = path_to_model_weights
        self.num_sequences = num_sequences
        self.omit_AAs = omit_AAs
        self.temperature = temperature
        self.design_selector = design_selector
        self.chains_to_mask = chains_to_mask
        self.mpnn_subtype = mpnn_subtype
        # setup standard command line flags for MPNN with default values
        self.flags = {
            "--backbone_noise": "0.0",
            "--max_length": "20000",
        }
        # add the flags that are required by MPNN and provided by the user
        self.flags.update(
            {
                "--batch_size": str(self.batch_size),
                "--num_seq_per_target": str(self.num_sequences),
                "--model_name": self.model_name,
                "--path_to_model_weights": self.path_to_model_weights,
                "--omit_AAs": self.omit_AAs,
                "--sampling_temp": str(self.temperature),
            }
        )
        self.allowed_flags = set([
            # flags that have default values that are provided by the runner:
            "--backbone_noise",
            "--max_length",
            # flags that are set by MPNNRunner constructor:
            "--batch_size",
            "--num_seq_per_target",
            "--path_to_model_weights",
            "--model_name",
            "--omit_AAs",
            "--sampling_temp",
            # flags that are required and are set by MPNNRunner or children:
            "--jsonl_path",
            "--chain_id_jsonl",
            "--fixed_positions_jsonl",
            "--out_folder",
            # flags that are optional and are set by MPNNRunner or children:
            "--assume_symmetry",
            "--bias_AA_jsonl",
            "--bias_by_res_jsonl",
            "--compute_input_sequence_score",
            "--omit_AA_jsonl",
            "--tied_positions_jsonl",
            "--pdb_path",
            "--pdb_path_chains",
            "--pssm_bias_flag",
            "--pssm_jsonl",
            "--pssm_log_odds_flag",
            "--pssm_multi",
            "--pssm_threshold",
            "--save_probs",
            "--save_score",
            "--seed",
            "--use_seed",
        ])
        if self.mpnn_subtype == "vanilla":
            self.script = f"{str(Path(__file__).resolve().parent.parent.parent / 'proteinmpnn' / 'protein_mpnn_run.py')}"
        else:
            self.script = f"{str(Path(__file__).resolve().parent.parent.parent / 'proteinmpnn' / self.mpnn_subtype / 'protein_mpnn_run.py')}"
            self.flags.pop("--path_to_model_weights")
            self.flags.pop("--model_name")
            self.allowed_flags.remove("--path_to_model_weights")
            self.allowed_flags.remove("--model_name")
        self.tmpdir = None  # this will be updated by the setup_tmpdir method.
        self.is_setup = False  # this will be updated by the setup_runner method.

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

    def setup_tmpdir(self) -> None:
        """
        :return: None
        Create a temporary directory for the MPNNRunner. Checks for various best
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
        Remove the temporary directory for the MPNNRunner.
        """
        import shutil

        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)
        return

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

    def setup_runner(self, pose: Pose, params: Optional[str] = None) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :param: params: optional path ligand parameters file if running LigandMPNN.
        :return: None
        Setup the MPNNRunner. Make a tmpdir and write input files to the tmpdir.
        Output sequences and scores will be written temporarily to the tmpdir as well.
        """
        import json
        import os
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        import git
        import pyrosetta
        import pyrosetta.distributed.io as io

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.utils.io import cmd

        # setup the tmpdir
        self.setup_tmpdir()
        # check that the first residue is on chain A, if not we should try to fix it
        if pose.pdb_info().chain(1) != "A":
            # ensure the chain numbers are correct by using SwitchChainOrderMover
            sc = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
            sc.chain_order("".join([str(c) for c in range(1, pose.num_chains() + 1)]))
            sc.apply(pose)
        else:
            pass
        # write the pose to a clean PDB file of only ATOM coordinates.
        tmp_pdb_path = os.path.join(self.tmpdir, "tmp.pdb")
        pdbstring = io.to_pdbstring(pose)
        with open(tmp_pdb_path, "w") as f:
            f.write(pdbstring)
        # make the jsonl file for the PDB biounits
        biounit_path = os.path.join(self.tmpdir, "biounits.jsonl")
        # use git to find the root of the repo
        repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        python = str(Path(root) / "envs" / "crispy" / "bin" / "python")
        if os.path.exists(python):
            pass
        else:  # crispy env must be installed in envs/crispy or must be used on DIGS
            python = "/projects/crispy_shifty/envs/shifty/bin/python"
        if self.mpnn_subtype == "vanilla":
            parser_script = str(
                Path(__file__).resolve().parent.parent.parent
                / "proteinmpnn"
                / "helper_scripts"
                / "parse_multiple_chains.py"
            )
        else:
            parser_script = str(
                Path(__file__).resolve().parent.parent.parent
                / "proteinmpnn"
                / self.mpnn_subtype
                / "helper_scripts"
                / "parse_multiple_chains.py"
            )
        if params is None:
            # continue if no ligand params file is provided
            pass
        else:
            # need to copy over the ligand params file to the tmpdir
            tmp_params_path = os.path.join(self.tmpdir, "tmp.params")
            shutil.copy(params, tmp_params_path)
            # need to change the helper script as well
            parser_script = parser_script.replace(
                "proteinmpnn", "proteinmpnn/ligand_proteinmpnn"
            )

        run_cmd = " ".join(
            [
                f"{python} {parser_script}",
                f"--input_path {self.tmpdir}",
                f"--output_path {biounit_path}",
            ]
        )
        out_err = cmd(run_cmd)
        print(out_err)
        # make a number to letter dictionary that starts at 1
        num_to_letter = {
            i: chr(i - 1 + ord("A")) for i in range(1, pose.num_chains() + 1)
        }
        # make the jsonl file for the chain_ids
        chain_id_path = os.path.join(self.tmpdir, "chain_id.jsonl")
        chain_dict = {}
        # make lists of masked and visible chains
        masked, visible = [], []
        # first make a list of all chain letters in the pose
        all_chains = [num_to_letter[i] for i in range(1, pose.num_chains() + 1)]
        # if chains_to_mask is provided, update the masked and visible lists
        if self.chains_to_mask is not None:
            # loop over the chains in the pose and add them to the appropriate list
            for chain in all_chains:
                if chain in self.chains_to_mask:
                    masked.append(i)
                else:
                    visible.append(i)
        else:
            # if chains_to_mask is not provided, mask all chains
            masked = all_chains
        chain_dict["tmp"] = [masked, visible]
        # write the chain_dict to a jsonl file
        with open(chain_id_path, "w") as f:
            f.write(json.dumps(chain_dict))
        # make the jsonl file for the fixed_positions
        fixed_positions_path = os.path.join(self.tmpdir, "fixed_positions.jsonl")
        fixed_positions_dict = {"tmp": {chain: [] for chain in all_chains}}
        # get a boolean mask of the residues in the design_selector
        if self.design_selector is not None:
            designable_filter = list(self.design_selector.apply(pose))
        else:  # if no design_selector is provided, make all residues designable
            designable_filter = [True] * pose.size()
        # check the residue design_selector specifies designability across the entire pose
        try:
            assert len(designable_filter) == pose.total_residue()
        except AssertionError:
            print(
                "Residue selector must specify designability for all residues.\n",
                f"Selector: {len(list(self.design_selector.apply(pose)))}\n",
                f"Pose: {pose.size()}",
            )
            raise
        # make a dict mapping of residue numbers to whether they are designable
        designable_dict = dict(zip(range(1, pose.size() + 1), designable_filter))
        # loop over the chains and the residues in the pose
        i = 1  # total residue counter
        for chain_number, chain in enumerate(all_chains, start=1):
            j = 1  # chain residue counter
            for res in range(
                pose.chain_begin(chain_number), pose.chain_end(chain_number) + 1
            ):
                # if the residue is on a masked chain but not designable, add it
                if not designable_dict[i] and chain in masked:
                    fixed_positions_dict["tmp"][chain].append(j)
                else:
                    pass
                j += 1
                i += 1
        # write the fixed_positions_dict to a jsonl file
        with open(fixed_positions_path, "w") as f:
            f.write(json.dumps(fixed_positions_dict))
        # update the flags for the biounit, chain_id, and fixed_positions paths
        flag_update = {
            "--jsonl_path": biounit_path,
            "--chain_id_jsonl": chain_id_path,
            "--fixed_positions_jsonl": fixed_positions_path,
        }
        self.update_flags(flag_update)
        self.is_setup = True
        return

    @abstractmethod
    def apply(self) -> None:
        """
        This function needs to be implemented by the child class of MPNNRunner.
        """
        pass


class MPNNDesign(MPNNRunner):
    """
    Class for running MPNN on a single interface selection or chain.
    TODO either in here or base class determine how to go from task factory to omit_AA_jsonl
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        :param: args: arguments to pass to MPNNRunner.
        :param: kwargs: keyword arguments to pass to MPNNRunner.
        Initialize the base class for MPNN runners with common attributes.
        """
        super().__init__(*args, **kwargs)

    def apply(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :return: None
        Run MPNN on the provided pose.
        Setup the MPNNRunner using the provided pose.
        Run MPNN in a subprocess using the provided flags and tmpdir.
        Read in and parse the output fasta file to get the sequences.
        Each sequence designed by MPNN is then appended to the pose datacache.
        Cleanup the tmpdir.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        import git
        import pyrosetta
        import pyrosetta.distributed.io as io
        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import fasta_to_dict, thread_full_sequence
        from crispy_shifty.utils.io import cmd

        # setup runner
        self.setup_runner(pose)
        self.update_flags({"--out_folder": self.tmpdir})

        # run mpnn by calling self.script and providing the flags
        # use git to find the root of the repo
        repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        python = str(Path(root) / "envs" / "crispy" / "bin" / "python")
        if os.path.exists(python):
            pass
        else:  # crispy env must be installed in envs/crispy or must be used on DIGS
            python = "/projects/crispy_shifty/envs/shifty/bin/python"
        run_cmd = (
            f"{python} {self.script}"
            + " "
            + " ".join([f"{k} {v}" for k, v in self.flags.items()])
        )
        print("Running MPNN with the following command:")
        print(run_cmd)
        out_err = cmd(run_cmd)
        print(out_err)
        alignments_path = os.path.join(self.tmpdir, "seqs/tmp.fa")
        # parse the alignments fasta into a dictionary
        alignments = fasta_to_dict(alignments_path, new_tags=True)
        for i, (tag, seq) in enumerate(alignments.items()):
            index = str(i).zfill(4)
            setPoseExtraScore(pose, f"mpnn_seq_{index}", seq)
        # clean up the temporary files
        self.teardown_tmpdir()
        return

    def dump_fasta(self, pose: Pose, out_path: str) -> None:
        """
        :param: pose: Pose object that contains the designed sequences.
        :param: out_path: Path to write the fasta file to.
        :return: None
        Dump the pose mpnn_seq_* sequences to a single fasta.
        """
        import sys
        from pathlib import Path

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import dict_to_fasta

        # get the mpnn_seq_* sequences from the pose
        seqs_to_write = {
            tag: seq for tag, seq in pose.scores.items() if "mpnn_seq_" in tag
        }
        # write the sequences to a fasta
        dict_to_fasta(seqs_to_write, out_path)

        return

    def generate_all_poses(
        self, pose: Pose, include_native: bool = False
    ) -> Iterator[Pose]:
        """
        :param: pose: Pose object to generate poses from.
        :param: include_native: Whether to generate the original native sequence.
        :return: Iterator of Pose objects.
        Generate poses from the provided pose with the newly designed sequences.
        Maintain the scores of the provided pose in the new poses.
        """
        import sys
        from pathlib import Path

        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import thread_full_sequence

        # get the mpnn_seq_* sequences from the pose
        seqs_dict = {tag: seq for tag, seq in pose.scores.items() if "mpnn_seq_" in tag}
        # get the non-mpnn_seq_* scores from the pose
        scores_dict = {
            key: val for key, val in pose.scores.items() if "mpnn_seq_" not in key
        }
        # generate the poses from the seqs_dict
        for tag, seq in seqs_dict.items():
            if include_native:
                pass
            else:  # don't include the original pose
                if tag == "mpnn_seq_0000":
                    continue
                else:
                    pass
            # thread the full sequence
            threaded_pose = thread_full_sequence(pose, seq)
            # set the scores
            for key, val in scores_dict.items():
                setPoseExtraScore(threaded_pose, key, val)
            yield threaded_pose


class MPNNLigandDesign(MPNNDesign):
    """
    Class for running MPNN on a ligand containing pose.
    """

    def __init__(
        self,
        *args,
        checkpoint_path: str = None,
        params: str = None,
        **kwargs,
    ):
        """
        :param: args: arguments to pass to MPNNRunner.
        :param: params: Path to the params file to use.
        :param: kwargs: keyword arguments to pass to MPNNRunner.
        Initialize the base class for MPNN runners with common attributes.
        Update the flags to use the provided params file.
        """
        super().__init__(*args, **kwargs)
        self.params = params
        # ensure the params file is provided
        if self.params is None:
            raise ValueError("Must provide params file.")
        else:
            pass
        # add allowed flags for the older inference script
        self.allowed_flags.update(["--use_ligand", "--checkpoint_path"])
        # take out model_name from the allowed flags
        self.allowed_flags.remove("--model_name")
        # change the script to the older script
        self.script = self.script.replace(
            "/protein_mpnn_run.py", "/ligand_proteinmpnn/protein_mpnn_run.py"
        )
        # set default checkpoint_path if not provided
        if checkpoint_path is None:
            self.checkpoint_path = self.script.replace(
                "/protein_mpnn_run.py",
                "/model_weights/lnet_plus10_010/epoch2000_step219536.pt",
            )
        else:
            self.checkpoint_path = checkpoint_path

        # update the flags
        self.update_flags({"--checkpoint_path": self.checkpoint_path})
        # remove model_name from the flags
        self.flags.pop("--model_name")
        # remove path_to_model_weights from the flags
        self.flags.pop("--path_to_model_weights")

    def apply(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :return: None
        Run MPNN on the provided pose.
        Setup the MPNNRunner using the provided pose.
        Run MPNN in a subprocess using the provided flags and tmpdir.
        Read in and parse the output fasta file to get the sequences.
        Each sequence designed by MPNN is then appended to the pose datacache.
        Cleanup the tmpdir.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        import git
        import pyrosetta
        import pyrosetta.distributed.io as io
        from pyrosetta.rosetta.core.pose import setPoseExtraScore

        # insert the root of the repo into the sys.path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from crispy_shifty.protocols.mpnn import fasta_to_dict, thread_full_sequence
        from crispy_shifty.utils.io import cmd

        # setup runner
        self.setup_runner(pose, params=self.params)
        self.update_flags({"--out_folder": self.tmpdir})

        # run mpnn by calling self.script and providing the flags
        # use git to find the root of the repo
        repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
        root = repo.git.rev_parse("--show-toplevel")
        python = str(Path(root) / "envs" / "crispy" / "bin" / "python")
        if os.path.exists(python):
            pass
        else:  # crispy env must be installed in envs/crispy or must be used on DIGS
            python = "/projects/crispy_shifty/envs/shifty/bin/python"
        # hacky way to get around confusing MPNN
        self.flags.pop("--chain_id_jsonl")
        run_cmd = (
            f"{python} {self.script}"
            + " "
            + " ".join([f"{k} {v}" for k, v in self.flags.items()])
        )
        out_err = cmd(run_cmd)
        print(out_err)
        alignments_path = os.path.join(self.tmpdir, "seqs/tmp.fa")
        # parse the alignments fasta into a dictionary
        alignments = fasta_to_dict(alignments_path, new_tags=True)
        for i, (tag, seq) in enumerate(alignments.items()):
            index = str(i).zfill(4)
            setPoseExtraScore(pose, f"mpnn_seq_{index}", seq)
        # clean up the temporary files
        self.teardown_tmpdir()
        return


class MPNNMultistateDesign(MPNNDesign):
    """
    Class for running MPNN to do multistate or sequence-symmetric design.
    """

    def __init__(
        self,
        residue_selectors: List[List[ResidueSelector]],
        residue_betas: Optional[List[List[float]]] = None,
        *args,
        **kwargs,
    ):
        """
        :param: args: arguments to pass to MPNNDesign.
        :param: kwargs: keyword arguments to pass to MPNNDesign.
        Initialize the base class for MPNN runners with common attributes.
        """
        super().__init__(*args, **kwargs)
        self.residue_selectors = residue_selectors
        self.residue_betas = residue_betas

    def setup_runner(self, pose: Pose) -> None:
        """
        :param: pose: Pose object to run MPNN on.
        :return: None
        Setup the MPNNRunner. Make a tmpdir and write input files to the tmpdir.
        Output sequences and scores will be written temporarily to the tmpdir as well.
        """
        import json
        import os
        from collections import defaultdict

        # take advantage of the superclass's setup_runner() to do most of the work
        super().setup_runner(pose)
        self.is_setup = False
        # make the jsonl file for the tied postions
        tied_positions_path = os.path.join(self.tmpdir, "tied_positions.jsonl")
        # set up residue_betas if not passed
        if self.residue_betas is None:
            self.residue_betas = []
            for sel_list in self.residue_selectors:
                self.residue_betas.append([1.0] * len(sel_list))
        # make a dict keyed by pose residue indices with chains as values
        chains_dict = {
            i: pose.pdb_info().chain(i) for i in range(1, pose.total_residue() + 1)
        }
        # make a dict mapping chain letters back to numbers
        chain_numbers = {
            letter: number
            for number, letter in enumerate(sorted(set(chains_dict.values())), start=1)
        }
        # get a boolean mask of the residues in the design_selector
        if self.design_selector is not None:
            designable_filter = list(self.design_selector.apply(pose))
        else:  # if no design_selector is provided, make all residues designable
            designable_filter = [True] * pose.size()
        # make a dict mapping of residue numbers to whether they are designable
        designable_dict = dict(zip(range(1, pose.size() + 1), designable_filter))
        tied_positions_dict = {"tmp": []}
        # now need the one indexed indices of all the True residues in the residue_selectors
        for sel_list, beta_list in zip(self.residue_selectors, self.residue_betas):
            residue_indices_lists = []
            for sel, beta in zip(sel_list, beta_list):
                residue_indices_list = []
                for i, selected in enumerate(sel.apply(pose), start=1):
                    if selected:
                        residue_indices_list.append((i, beta))
                    else:
                        pass
                residue_indices_lists.append(residue_indices_list)
            # flatten the list of lists into a single list of tuples by unpacking the above
            tied_positions_list = list(zip(*residue_indices_lists))
            for tied_positions in tied_positions_list:
                # get the chains for the tied positions
                tied_position_dict = defaultdict(lambda: [[], []])
                # we only tie a position if all the residues are up for design
                designable = True
                for tied_position, residue_beta in tied_positions:
                    # check if the residue is designable
                    if designable_dict[tied_position]:
                        pass
                    else:
                        designable = False
                    # get the residue index and chain
                    chain = chains_dict[tied_position]
                    # we need to offset the residue index by the chain now
                    residue_index = (
                        tied_position - pose.chain_begin(chain_numbers[chain]) + 1
                    )
                    # add the residue index and chain to the dict
                    tied_position_dict[chain][0].append(residue_index)
                    tied_position_dict[chain][1].append(residue_beta)
                # skip this tied position if any of the residues are not designable
                if not designable:
                    continue
                else:
                    pass
                # the output json should have a single entry with a list of dicts of the tied positions
                tied_positions_dict["tmp"].append(dict(tied_position_dict))

        # write the tied_positions_dict to a jsonl file
        with open(tied_positions_path, "w") as f:
            f.write(json.dumps(tied_positions_dict))
        # update the flags for the biounit, chain_id, and fixed_positions paths
        flag_update = {
            "--tied_positions_jsonl": tied_positions_path,
        }
        self.update_flags(flag_update)
        self.is_setup = True
        return


@requires_init
def mpnn_bound_state(
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
    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    interface_selector = interface_between_selectors(chA, chB)
    neighborhood_selector = NeighborhoodResidueSelector(
        interface_selector, distance=8.0, include_focus_in_subset=True
    )
    full_selector = TrueResidueSelector()
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
            design_sel = (
                pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(
                    mpnn_design_area,
                    pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(
                        fixed_sel
                    ),
                )
            )
            print_timestamp("Designing interface with MPNN", start_time)
            # construct the MPNNDesign object
            mpnn_design = MPNNDesign(
                design_selector=design_sel,
                omit_AAs="CX",
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
def mpnn_paired_state(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNMultistateDesign, or this function.
    :return: an iterator of PackedPose objects.
    """

    import sys
    from itertools import product
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        NeighborhoodResidueSelector,
        OrResidueSelector,
        ResidueIndexSelector,
        TrueResidueSelector,
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
    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    chC = ChainSelector(3)
    interface_selector = interface_between_selectors(chA, chB)
    neighborhood_selector = NeighborhoodResidueSelector(
        interface_selector, distance=8.0, include_focus_in_subset=True
    )
    full_selector = TrueResidueSelector()
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

    # make a list of linked residue selectors, we want to link chA and chC
    residue_selectors = [[chA, chC]]

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        # get the length of state Y
        offset = pose.chain_end(2)
        # get a boolean mask of the residues in chA
        chA_filter = list(chA.apply(pose))
        # make a list of the corresponding residues in state X that are interface in Y
        X_interface_residues = [
            i + offset for i, designable in enumerate(chA_filter, start=1) if designable
        ]
        X_interface_residues_str = ",".join(str(i) for i in X_interface_residues)
        X_selector = ResidueIndexSelector(X_interface_residues_str)
        # iterate over the mpnn parameter combinations
        mpnn_conditions = list(product(mpnn_temperatures, mpnn_design_areas))
        num_conditions = len(list(mpnn_conditions))
        print_timestamp(f"Beginning {num_conditions} MPNNDesign runs", start_time)
        for i, (mpnn_temperature, mpnn_design_area) in enumerate(list(mpnn_conditions)):
            pose = original_pose.clone()
            design_selector = OrResidueSelector(mpnn_design_area, X_selector)
            print_timestamp(
                f"Beginning MPNNDesign run {i+1}/{num_conditions}", start_time
            )

            print_timestamp("Multistate design with MPNN", start_time)
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
            scores.update(
                {
                    "mpnn_msd_temperature": mpnn_temperature,
                    "mpnn_msd_design_area": selector_inverse_options[mpnn_design_area],
                }
            )
            # update the pose with the updated scores dict
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            # generate the original pose, with the sequences written to the datacache
            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def mpnn_paired_state_fixed(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNMultistateDesign, or this function.
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
        FalseResidueSelector,
        NeighborhoodResidueSelector,
        NotResidueSelector,
        OrResidueSelector,
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
        cluster_scores = kwargs.pop("df_scores", True)
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=cluster_scores, pack_result=False
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
    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    chC = ChainSelector(3)
    interface_selector = interface_between_selectors(chA, chB)
    neighborhood_selector = NeighborhoodResidueSelector(
        interface_selector, distance=8.0, include_focus_in_subset=True
    )
    # full_selector = TrueResidueSelector()
    full_selector = OrResidueSelector(chA, chB)
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

    # make a list of linked residue selectors, we want to link chA and chC
    residue_selectors = [[chA, chC]]

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        # don't design any fixed residues
        fixed_sel = FalseResidueSelector()
        if "fixed_resis" in scores:
            fixed_resi_str = scores["fixed_resis"]
            # handle an empty string
            if fixed_resi_str:
                fixed_sel = ResidueIndexSelector(fixed_resi_str)
        original_pose = pose.clone()
        # get the length of state Y
        offset = pose.chain_end(2)
        # get the start of the peptide
        peptide_start = pose.chain_end(3) + 1
        # iterate over the mpnn parameter combinations
        mpnn_conditions = list(product(mpnn_temperatures, mpnn_design_areas))
        num_conditions = len(list(mpnn_conditions))
        print_timestamp(f"Beginning {num_conditions} MPNNDesign runs", start_time)
        for i, (mpnn_temperature, mpnn_design_area) in enumerate(list(mpnn_conditions)):
            pose = original_pose.clone()
            # get a boolean mask of the designable residues in state Y
            Y_design_sel = AndResidueSelector(
                mpnn_design_area, NotResidueSelector(fixed_sel)
            )
            Y_designable_filter = list(Y_design_sel.apply(pose))
            Y_designable_residues = [
                i
                for i, designable in enumerate(Y_designable_filter, start=1)
                if designable
            ]
            # make a list of the corresponding residues in state X that are designable in Y
            X_designable_residues = [
                i + offset for i in Y_designable_residues if i + offset < peptide_start
            ]
            designable_residues = Y_designable_residues + X_designable_residues
            design_sel = ResidueIndexSelector(
                ",".join(str(i) for i in designable_residues)
            )
            print_timestamp(
                f"Beginning MPNNDesign run {i+1}/{num_conditions}", start_time
            )

            print_timestamp("Multistate design with MPNN", start_time)
            # construct the MPNNMultistateDesign object
            mpnn_design = MPNNMultistateDesign(
                design_selector=design_sel,
                residue_selectors=residue_selectors,
                omit_AAs="CX",
                **kwargs,
            )
            # design the pose
            mpnn_design.apply(pose)
            print_timestamp("MPNN design complete, updating pose datacache", start_time)
            # update the scores dict
            scores.update(pose.scores)
            scores.update(
                {
                    "mpnn_msd_temperature": mpnn_temperature,
                    "mpnn_msd_design_area": selector_inverse_options[mpnn_design_area],
                }
            )
            # update the pose with the updated scores dict
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            # generate the original pose, with the sequences written to the datacache
            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def mpnn_tie_all(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNMultistateDesign, or this function.
    :return: an iterator of PackedPose objects.
    Runs MPNN design, tying residues of all chain sets in the input pose. All corresponding 
    chains among sets must be the same length.
    Right now you can assign betas to each chain unit, but I haven't implemented a way to assign
    different betas to chains within a unit.
    """

    import sys
    from itertools import product
    import numpy as np
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        NeighborhoodResidueSelector,
        ResidueIndexSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import interface_between_selectors, interface_among_chains, selector_union, selector_intersection
    from crispy_shifty.protocols.mpnn import MPNNMultistateDesign
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        cluster_scores = kwargs.pop("df_scores", True)
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=cluster_scores, pack_result=False
        )

    chains_per_unit = int(kwargs.pop("chains_per_unit", 1))
        
    generate_all_poses = kwargs.pop("generate_all_poses", False)
    if generate_all_poses and generate_all_poses.lower() == "true":
        generate_all_poses = True

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
        mpnn_temperatures = [0.2]

    # a list of lists containing beta values for each chain pair
    if "mpnn_betas" in kwargs:
        mpnn_betas_list = [[float(beta) for beta in beta_list.split(',')] for beta_list in kwargs["mpnn_betas"].split(';')]
    else:
        mpnn_betas_list = None

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()

        num_chains = pose.num_chains()
        num_units = num_chains // chains_per_unit

        # there should be a beta value corresponding to each chain
        if mpnn_betas_list:
            for mpnn_betas in mpnn_betas_list:
                assert len(mpnn_betas) == num_units
        else:
            mpnn_betas_list = [[1 / num_units] * num_units]
        print(mpnn_betas_list)

        # setup dict for MPNN design areas
        print_timestamp("Setting up design selectors", start_time)
        # make designable residue selectors
        chain_sels = [ChainSelector(i) for i in range(1, num_chains + 1)]
        intraunit_interface_selectors = [interface_among_chains(list(range(i, i+chains_per_unit)), vector_mode=True) for i in range(1, num_chains+1, chains_per_unit)]
        intraunit_neighborhood_selectors = [
            NeighborhoodResidueSelector(
                interface_selector, distance=8.0, include_focus_in_subset=True
            )
            for interface_selector in intraunit_interface_selectors
        ]
        full_selectors = [selector_union(chain_sels[i:i+chains_per_unit]) for i in range(0, num_chains, chains_per_unit)]
        # selects the interface between the first and all other asymmetric units
        # in a symmetric structure, assumes that all the other interfaces involve the same set of residues in each unit
        interunit_sym_interface_sel = \
            selector_union([
                selector_intersection([
                    full_selectors[0],
                    interface_between_selectors(full_selectors[0], full_selectors[i], vector_mode=True)
                ])
                for i in range(1, num_units)
            ])
        interunit_sym_neighborhood_sel = \
            selector_intersection([
                full_selectors[0],
                NeighborhoodResidueSelector(
                    interunit_sym_interface_sel, distance=8.0, include_focus_in_subset=True
                )
            ])
        selector_options = {
            "full": full_selectors,
            "intraunit_interface": intraunit_interface_selectors,
            "intraunit_neighborhood": intraunit_neighborhood_selectors,
            "interunit_sym_interface": [interunit_sym_interface_sel],
            "interunit_sym_neighborhood": [interunit_sym_neighborhood_sel]
        }
        if "mpnn_design_area" in kwargs:
            if kwargs["mpnn_design_area"] == "scan":
                mpnn_design_areas = [
                    (key, selector_options[key])
                    for key in ["full", "intraunit_interface", "intraunit_neighborhood"]
                ]
            elif kwargs["mpnn_design_area"] == "scan_sym":
                mpnn_design_areas = [
                    (key, selector_options[key])
                    for key in ["full", "interunit_sym_interface", "interunit_sym_neighborhood"]
                ]
            else:
                try:
                    key = kwargs["mpnn_design_area"]
                    mpnn_design_areas = [(key, selector_options[key])]
                except:
                    raise ValueError(
                        "mpnn_design_area must be one of the following: full, intraunit_interface, intraunit_neighborhood, interunit_sym_interface, interunit_sym_neighborhood"
                    )
        else:
            mpnn_design_areas = [("full", selector_options["full"])]
        # make a list of linked residue selectors
        residue_selectors = [chain_sels[i::chains_per_unit] for i in range(num_units)]

        # get the length of one chain set
        len_unit = pose.chain_end(chains_per_unit)
        # iterate over the mpnn parameter combinations
        mpnn_conditions = product(mpnn_temperatures, mpnn_design_areas, mpnn_betas_list)
        print_timestamp(f"Beginning MPNNDesign runs", start_time)
        for run_i, (mpnn_temperature, mpnn_design_area, mpnn_betas) in enumerate(mpnn_conditions):
            pose = original_pose.clone()

            # get a mask of all the residues that are designable in any chain unit
            unit_mask = np.zeros(len_unit, dtype=bool)
            for i, unit_sel in enumerate(mpnn_design_area[1]):
                unit_mask = np.logical_or(
                    unit_mask,
                    np.array(
                        list(unit_sel.apply(pose))[i*len_unit:(i+1)*len_unit]
                    )
                )
            # build a list of all residue numbers that are designable from the mask of residues that are designable in any chain unit
            all_interface_residues = []
            for i in range(num_units):
                all_interface_residues += [
                    i*len_unit + j
                    for j, designable in enumerate(unit_mask, start=1)
                    if designable
                ]
            all_interface_residues_str = ",".join(
                str(i) for i in all_interface_residues
            )
            print(all_interface_residues_str)
            design_selector = ResidueIndexSelector(all_interface_residues_str)

            print_timestamp(
                f"Beginning MPNNDesign run {run_i+1}", start_time
            )
            # construct the MPNNMultistateDesign object
            mpnn_design = MPNNMultistateDesign(
                design_selector=design_selector,
                residue_selectors=residue_selectors,
                residue_betas=[mpnn_betas] * chains_per_unit,
                **kwargs,
            )
            # design the pose
            mpnn_design.apply(pose)
            # update the scores dict
            if generate_all_poses:
                out_poses = mpnn_design.generate_all_poses(pose, include_native=True)
            else:
                out_poses = [pose]
            for out_pose in out_poses:
                scores.update(out_pose.scores)
                scores.update(
                    {
                        "mpnn_msd_temperature": mpnn_temperature,
                        "mpnn_msd_design_area": mpnn_design_area[0],
                        "mpnn_msd_betas": ",".join(str(beta) for beta in mpnn_betas),
                    }
                )
                # update the pose with the updated scores dict
                for key, value in scores.items():
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(out_pose, key, value)
                # generate the original pose, with the sequences written to the datacache
                ppose = io.to_packed(out_pose)
                yield ppose

@requires_init
def mpnn_selection(
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
        ResidueIndexSelector,
        NeighborhoodResidueSelector,
        TrueResidueSelector,
        FalseResidueSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.mpnn import MPNNDesign
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    cluster_scores = kwargs.pop("df_scores", True)

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=cluster_scores, pack_result=False
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
        
    generate_all_poses = kwargs.pop("generate_all_poses", False)
    if generate_all_poses and generate_all_poses.lower() == "true":
        generate_all_poses = True

    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the selected residues
    selection_selector = ResidueIndexSelector()
    neighborhood_selector = NeighborhoodResidueSelector(
        selection_selector, distance=8.0, include_focus_in_subset=True
    )
    full_selector = TrueResidueSelector()
    selector_options = {
        "selection": selection_selector,
        "neighborhood": neighborhood_selector,
        "full": full_selector,
    }
    # make the inverse dict of selector options
    selector_inverse_options = {value: key for key, value in selector_options.items()}
    if "mpnn_design_area" in kwargs:
        if kwargs["mpnn_design_area"] == "scan":
            mpnn_design_areas = [
                selector_options[key] for key in ["selection", "neighborhood", "full"]
            ]
        else:
            try:
                mpnn_design_areas = [selector_options[kwargs["mpnn_design_area"]]]
            except:
                raise ValueError(
                    "mpnn_design_area must be one of the following: selection, neighborhood, full"
                )
    else:
        mpnn_design_areas = [selector_options["selection"]]

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)

        if kwargs["mpnn_design_area"] != "full":
            selection_selector.set_index(scores[kwargs["selection_name"]])

        # don't design any fixed residues
        fixed_sel = FalseResidueSelector()
        if "fixed_resis" in scores:
            fixed_resi_str = scores["fixed_resis"]
            # handle an empty string
            if fixed_resi_str:
                fixed_sel = ResidueIndexSelector(fixed_resi_str)
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
            design_sel = (
                pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(
                    mpnn_design_area,
                    pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(
                        fixed_sel
                    ),
                )
            )
            print_timestamp("Designing interface with MPNN", start_time)
            # construct the MPNNDesign object
            mpnn_design = MPNNDesign(
                design_selector=design_sel,
                temperature=mpnn_temperature,
                **kwargs,
            )
            # design the pose
            mpnn_design.apply(pose)
            print_timestamp("MPNN design complete, updating pose datacache", start_time)
            # update the scores dict
            if generate_all_poses:
                out_poses = mpnn_design.generate_all_poses(pose, include_native=True)
            else:
                out_poses = [pose]
            for out_pose in out_poses:
                scores.update(out_pose.scores)
                scores.update(
                    {
                        "mpnn_temperature": mpnn_temperature,
                        "mpnn_design_area": selector_inverse_options[mpnn_design_area],
                    }
                )
                # update the pose with the updated scores dict
                for key, value in scores.items():
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(out_pose, key, value)
                # generate the original pose, with the sequences written to the datacache
                ppose = io.to_packed(out_pose)
                yield ppose

@requires_init
def mpnn_fastrelax_selection(
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
        ResidueIndexSelector,
        NeighborhoodResidueSelector,
        ChainSelector,
        TrueResidueSelector,
        FalseResidueSelector,
        OrResidueSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        fast_relax,
        gen_std_layer_design,
        gen_task_factory,
        gen_scorefxn,
        gen_movemap,
        score_ddg,
        score_sc,
        score_cms,
    )
    from crispy_shifty.protocols.mpnn import MPNNDesign
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    cluster_scores = kwargs.pop("df_scores", True)
    if "mpnn_fr_cycles" in kwargs:
        mpnn_fr_cycles = int(kwargs["mpnn_fr_cycles"])
    else:
        mpnn_fr_cycles = 1
    if "cart_relax" in kwargs:
        cart_relax = bool(kwargs["cart_relax"])
    else:
        cart_relax = False

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=cluster_scores, pack_result=False
        )

    if "mpnn_temperature" in kwargs:
        if kwargs["mpnn_temperature"] == "scan":
            mpnn_temperatures = [0.0001, 0.1, 0.2, 0.5]
        else:
            mpnn_temperature = float(kwargs["mpnn_temperature"])
            assert (
                0.0 < mpnn_temperature <= 1.0
            ), "mpnn_temperature must be between 0 and 1"
            mpnn_temperatures = [mpnn_temperature]
    else:
        mpnn_temperatures = [0.0001]

    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the selected residues
    mpnn_selection_selector = ResidueIndexSelector()
    mpnn_neighborhood_selector = NeighborhoodResidueSelector(
        mpnn_selection_selector, distance=8.0, include_focus_in_subset=True
    )
    mpnn_full_selector = TrueResidueSelector()
    mpnn_selector_options = {
        "selection": mpnn_selection_selector,
        "neighborhood": mpnn_neighborhood_selector,
        "full": mpnn_full_selector,
    }
    # make the inverse dict of selector options
    mpnn_selector_inverse_options = {value: key for key, value in mpnn_selector_options.items()}
    if "mpnn_design_area" in kwargs:
        if kwargs["mpnn_design_area"] == "scan":
            mpnn_design_areas = [
                mpnn_selector_options[key] for key in ["selection", "neighborhood", "full"]
            ]
        else:
            try:
                mpnn_design_areas = [mpnn_selector_options[kwargs["mpnn_design_area"]]]
            except:
                raise ValueError(
                    "mpnn_design_area must be one of the following: selection, neighborhood, full"
                )
    else:
        mpnn_design_areas = [mpnn_selector_options["selection"]]

    sfxn = gen_scorefxn(cartesian=cart_relax)

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)

        if kwargs["mpnn_design_area"] != "full":
            mpnn_selection_selector.set_index(scores[kwargs["mpnn_selection_name"]])

        # don't design any fixed residues
        fixed_sel = FalseResidueSelector()
        if "fixed_resis" in scores:
            fixed_resi_str = scores["fixed_resis"]
            # handle an empty string
            if fixed_resi_str:
                fixed_sel = ResidueIndexSelector(fixed_resi_str)
        original_pose = pose.clone()
        # iterate over the mpnn parameter combinations
        mpnn_conditions = list(product(mpnn_temperatures, mpnn_design_areas))
        num_conditions = len(list(mpnn_conditions))
        print_timestamp(f"Beginning {num_conditions} iterated MPNNDesign/FastRelax runs", start_time)
        for i, (mpnn_temperature, mpnn_design_area) in enumerate(list(mpnn_conditions)):
            pose = original_pose.clone()
            print_timestamp(
                f"Beginning MPNNDesign/FastRelax run {i+1}/{num_conditions}", start_time
            )
            design_sel = (
                pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(
                    mpnn_design_area,
                    pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(
                        fixed_sel
                    ),
                )
            )
            task_factory = gen_task_factory(
                design_sel=design_sel,
                pack_nbhd=True,
                extra_rotamers_level=2,
                limit_arochi=True,
                include_current=True,
                upweight_ppi=True,
                # precompute_ig=True, # this was using like >14G of memory
            )
            if "relax_area" in kwargs:
                relax_area_name = kwargs["relax_area"]
            else:
                relax_area_name = "full"
            if relax_area_name == "full":
                relax_area = TrueResidueSelector()
            elif relax_area_name == "selection":
                relax_area = ResidueIndexSelector(scores[kwargs["relax_selection_name"]])
            elif relax_area_name == "neighborhood":
                relax_area = NeighborhoodResidueSelector(ResidueIndexSelector(scores[kwargs["relax_selection_name"]]), distance=8.0, include_focus_in_subset=True)
            elif relax_area_name == "design_chains":
                relax_area = OrResidueSelector()
                # allow backbone movements of chains which are being designed
                designed_chains = set()
                for resi in pyrosetta.rosetta.core.select.get_residues_from_subset(design_sel.apply(pose)):
                    designed_chains.add(pose.chain(resi))
                for chain in designed_chains:
                    chain_sel = ChainSelector(chain)
                    relax_area.add_residue_selector(chain_sel)
            else:
                raise ValueError(
                    "relax_area must be one of the following: selection, neighborhood, design_chains, full"
                )
            # print(f"relax_area: {relax_area_name}")
            # print(pyrosetta.rosetta.core.select.get_residues_from_subset(relax_area.apply(pose)))
            move_map = gen_movemap(jump=True, chi=False, bb=False)
            for resi in pyrosetta.rosetta.core.select.get_residues_from_subset(relax_area.apply(pose)):
                move_map.set_bb(resi, True)
                move_map.set_chi(resi, True)
            # construct the MPNNDesign object
            mpnn_design = MPNNDesign(
                design_selector=design_sel,
                temperature=mpnn_temperature,
                batch_size=1,
                num_sequences=1,
                **kwargs,
            )
            for i in range(mpnn_fr_cycles):
                print_timestamp(f"Iteration {i+1}/{mpnn_fr_cycles}", start_time)
                # design the pose
                mpnn_design.apply(pose)
                # only one sequence
                pose = next(mpnn_design.generate_all_poses(pose, include_native=False))
                # relax the pose
                fast_relax(
                    pose=pose,
                    task_factory=task_factory,
                    scorefxn=sfxn,
                    movemap=move_map,
                    relax_script="MonomerDesign2019",
                    repeats=1,
                    cartesian=cart_relax,
                )
            # final sequence design
            mpnn_design.apply(pose)
            pose = next(mpnn_design.generate_all_poses(pose, include_native=False))
            if pose.num_chains() == 2:
                ch1_sel = ChainSelector(1)
                ch2_sel = ChainSelector(2)
                score_cms(pose, ch1_sel, ch2_sel)
                score_sc(pose, ch1_sel, ch2_sel)
                score_ddg(pose)
            # update the scores dict
            scores.update(pose.scores)
            scores.update(
                {
                    "mpnn_temperature": mpnn_temperature,
                    "mpnn_design_area": mpnn_selector_inverse_options[mpnn_design_area],
                    "mpnn_fr_cycles": mpnn_fr_cycles,
                    "relax_area": relax_area_name,
                }
            )
            # update the pose with the updated scores dict
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            # generate the original pose, with the sequences written to the datacache
            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def mpnn_bound_state_flaggable(
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
    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    interface_selector = interface_between_selectors(chA, chB)
    neighborhood_selector = NeighborhoodResidueSelector(
        interface_selector, distance=8.0, include_focus_in_subset=True
    )
    full_selector = TrueResidueSelector()
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
            design_sel = (
                pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(
                    mpnn_design_area,
                    pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(
                        fixed_sel
                    ),
                )
            )
            print_timestamp("Designing interface with MPNN", start_time)
            # construct the MPNNDesign object
            mpnn_design = MPNNDesign(
                design_selector=design_sel,
                omit_AAs="CX",
                temperature=mpnn_temperature,
                **kwargs,
            )
            # design the pose
            mpnn_design.update_flags({f'--{k}':v for k,v in kwargs.items() if f'--{k}' in mpnn_design.allowed_flags})
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
def mpnn_paired_state_fixed_betas_flaggable(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNMultistateDesign, or this function.
    :return: an iterator of PackedPose objects.
    """

    from itertools import product
    from pathlib import Path
    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        AndResidueSelector,
        OrResidueSelector,
        NotResidueSelector,
        NeighborhoodResidueSelector,
        ResidueIndexSelector,
        FalseResidueSelector,
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

    if "mpnn_betas" in kwargs:
        if kwargs["mpnn_betas"] == "scan":
            mpnn_betas_list = [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]]#, [0.7, 0.3]]
        else:
            mpnn_betas_list = [[float(beta) for beta in beta_list.split(',')] for beta_list in kwargs["mpnn_betas"].split(';')]
    else:
        mpnn_betas_list = [[0.6, 0.4]]

    for mpnn_betas in mpnn_betas_list:
        assert len(mpnn_betas) == 2
    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    chC = ChainSelector(3)
    interface_selector = interface_between_selectors(chA, chB)
    neighborhood_selector = NeighborhoodResidueSelector(
        interface_selector, distance=8.0, include_focus_in_subset=True
    )
    # full_selector = TrueResidueSelector()
    full_selector = OrResidueSelector(chA, chB)
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

    # make a list of linked residue selectors, we want to link chA and chC
    residue_selectors = [[chA, chC]]

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        # don't design any fixed residues
        fixed_sel = FalseResidueSelector()
        if "fixed_resis" in scores:
            fixed_resi_str = scores["fixed_resis"]
            print(fixed_resi_str, flush=True)
            # handle an empty string
            if fixed_resi_str:
                fixed_sel = ResidueIndexSelector(fixed_resi_str)
        original_pose = pose.clone()
        # get the length of state Y
        offset = pose.chain_end(2)
        # get the start of the peptide
        peptide_start = pose.chain_end(3)+1
        # iterate over the mpnn parameter combinations
        mpnn_conditions = list(
            product(mpnn_temperatures, mpnn_design_areas, mpnn_betas_list)
        )
        num_conditions = len(list(mpnn_conditions))
        print_timestamp(f"Beginning {num_conditions} MPNNDesign runs", start_time)
        for i, (mpnn_temperature, mpnn_design_area, mpnn_betas) in enumerate(
            list(mpnn_conditions)
        ):
            pose = original_pose.clone()
            # get a boolean mask of the designable residues in state Y
            Y_design_sel = AndResidueSelector(
                mpnn_design_area, NotResidueSelector(fixed_sel)
            )
            Y_designable_filter = list(Y_design_sel.apply(pose))
            Y_designable_residues = [
                i
                for i, designable in enumerate(Y_designable_filter, start=1)
                if designable
            ]
            # make a list of the corresponding residues in state X that are designable in Y
            X_designable_residues = [
                i + offset for i in Y_designable_residues if i + offset < peptide_start
            ]
            designable_residues = Y_designable_residues + X_designable_residues
            design_sel = ResidueIndexSelector(
                ",".join(str(i) for i in designable_residues)
            )
            print_timestamp(
                f"Beginning MPNNDesign run {i+1}/{num_conditions}", start_time
            )

            print_timestamp("Multistate design with MPNN", start_time)
            # construct the MPNNMultistateDesign object
            mpnn_design = MPNNMultistateDesign(
                design_selector=design_sel,
                residue_selectors=residue_selectors,
                residue_betas=[mpnn_betas],
                temperature=mpnn_temperature,
                omit_AAs="CX",
                **kwargs,
            )
            # design the pose
            mpnn_design.update_flags({f'--{k}':v for k,v in kwargs.items() if f'--{k}' in mpnn_design.allowed_flags})
            mpnn_design.apply(pose)
            print_timestamp("MPNN design complete, updating pose datacache", start_time)
            # update the scores dict
            scores.update(pose.scores)
            scores.update(
                {
                    "mpnn_msd_temperature": mpnn_temperature,
                    "mpnn_msd_design_area": selector_inverse_options[mpnn_design_area],
                    "mpnn_msd_betas": ",".join(str(beta) for beta in mpnn_betas),
                }
            )
            # update the pose with the updated scores dict
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            # generate the original pose, with the sequences written to the datacache
            ppose = io.to_packed(pose)
            yield ppose


@requires_init
def mpnn_paired_state_arbitrary(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNMultistateDesign, or this function.
    :return: an iterator of PackedPose objects.
    """

    from itertools import product
    from pathlib import Path
    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        AndResidueSelector,
        OrResidueSelector,
        NotResidueSelector,
        NeighborhoodResidueSelector,
        ResidueIndexSelector,
        FalseResidueSelector,
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

    if "mpnn_betas" in kwargs:
        if kwargs["mpnn_betas"] == "scan":
            mpnn_betas_list = [[0.5, 0.5], [0.6, 0.4]]#, [0.7, 0.3]]
        else:
            mpnn_betas_list = [[float(beta) for beta in beta_list.split(',')] for beta_list in kwargs["mpnn_betas"].split(';')]
    else:
        mpnn_betas_list = [[0.6, 0.4]]

    for mpnn_betas in mpnn_betas_list:
        assert len(mpnn_betas) == 2
    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    chC = ChainSelector(3)
    interface_selector = interface_between_selectors(chA, chB)
    neighborhood_selector = NeighborhoodResidueSelector(
        interface_selector, distance=8.0, include_focus_in_subset=True
    )
    # full_selector = TrueResidueSelector()
    full_selector = OrResidueSelector(chA, chB)
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

    # make a list of linked residue selectors, we want to link chA and chC
    #residue_selectors = [[chA, chC]]
    if "sym_sele" in kwargs:
        sym_sele = kwargs["sym_sele"]
        sym_seleA=ResidueIndexSelector(sym_sele.split(';')[0])
        sym_seleC=ResidueIndexSelector(sym_sele.split(';')[-1])
        residue_selectors = [[sym_seleA, sym_seleC]]
    else:
        residue_selectors = [[chA, chC]]
    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        # don't design any fixed residues
        fixed_sel = FalseResidueSelector()
        if "fixed_resis" in scores:
            fixed_resi_str = scores["fixed_resis"]
            print(fixed_resi_str, flush=True)
            # handle an empty string
            if fixed_resi_str:
                fixed_sel = ResidueIndexSelector(fixed_resi_str)
        original_pose = pose.clone()
        # get the length of state Y
        offset = pose.chain_end(2)
        # get the start of the peptide
        peptide_start = pose.chain_end(3)+1
        # iterate over the mpnn parameter combinations
        mpnn_conditions = list(
            product(mpnn_temperatures, mpnn_design_areas, mpnn_betas_list)
        )
        num_conditions = len(list(mpnn_conditions))
        print_timestamp(f"Beginning {num_conditions} MPNNDesign runs", start_time)
        for i, (mpnn_temperature, mpnn_design_area, mpnn_betas) in enumerate(
            list(mpnn_conditions)
        ):
            pose = original_pose.clone()
            # get a boolean mask of the designable residues in state Y
            Y_design_sel = AndResidueSelector(
                mpnn_design_area, NotResidueSelector(fixed_sel)
            )
            Y_designable_filter = list(Y_design_sel.apply(pose))
            Y_designable_residues = [
                i
                for i, designable in enumerate(Y_designable_filter, start=1)
                if designable
            ]
            # make a list of the corresponding residues in state X that are designable in Y
            X_designable_residues = [
                i + offset for i in Y_designable_residues if i + offset < peptide_start
            ]
            designable_residues = Y_designable_residues + X_designable_residues
            design_sel = ResidueIndexSelector(
                ",".join(str(i) for i in designable_residues)
            )
            print_timestamp(
                f"Beginning MPNNDesign run {i+1}/{num_conditions}", start_time
            )

            print_timestamp("Multistate design with MPNN", start_time)
            # construct the MPNNMultistateDesign object
            mpnn_design = MPNNMultistateDesign(
                design_selector=design_sel,
                residue_selectors=residue_selectors,
                residue_betas=[mpnn_betas],
                omit_AAs="CX",
                **kwargs,
            )
            # design the pose
            mpnn_design.update_flags({f'--{k}':v for k,v in kwargs.items() if f'--{k}' in mpnn_design.allowed_flags})
            mpnn_design.apply(pose)
            print_timestamp("MPNN design complete, updating pose datacache", start_time)
            # update the scores dict
            scores.update(pose.scores)
            scores.update(
                {
                    "mpnn_msd_temperature": mpnn_temperature,
                    "mpnn_msd_design_area": selector_inverse_options[mpnn_design_area],
                    "mpnn_msd_betas": ",".join(str(beta) for beta in mpnn_betas),
                }
            )
            # update the pose with the updated scores dict
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
            # generate the original pose, with the sequences written to the datacache
            ppose = io.to_packed(pose)
            yield ppose
