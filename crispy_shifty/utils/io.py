# Python standard library
import bz2
import collections
import json
import os
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, NoReturn, Optional, Tuple, Union

# 3rd party library imports
import pandas as pd
import pyrosetta.distributed.io as io
import toolz

# Rosetta library imports
from pyrosetta.distributed import requires_init
from pyrosetta.distributed.cluster.exceptions import OutputError
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueSelector
from tqdm.auto import tqdm

# Custom library imports


def cmd(command: str = "", wait: bool = True) -> str:
    """
    :param: command: Command to run.
    :param: wait: Wait for command to finish.
    :return: stdout.
    Run a command.
    """
    import os
    import subprocess

    p = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if wait:
        out = str(p.communicate()[0]) + str(p.communicate()[1])
        return out
    else:
        return


def cmd_no_stderr(command: str = "", wait: bool = True) -> str:
    """
    :param: command: Command to run.
    :param: wait: Wait for command to finish.
    :return: stdout.
    Run a command and suppress stderr.
    """
    import os
    import subprocess

    with open(os.devnull, "w") as devnull:
        p = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=devnull,
            universal_newlines=True,
        )
    if wait:
        out = str(p.communicate()[0])
        return out
    else:
        return


def fix_path_prefixes(
    find: str, replace: str, file: str, overwrite: Optional[bool] = False
) -> Union[pd.DataFrame, List[str], str, None]:
    """
    :param: find: path prefix to find and replace.
    :param: replace: path prefix to replace with.
    :param: file: path to file to fix.
    :param: overwrite: overwrite file if it exists.
    :return: A pandas dataframe, a list of paths, or a string, depending on the file
    type of input, or None if overwrite is True.
    When rsyncing lists or scorefiles from one cluster to another, the paths in the
    lists or index of the scorefile need to be fixed. This function fixes them, and
    can either return a python object (dataframe, list, or string) of the fixed file,
    or overwrite the file in place.
    """
    import os

    import pandas as pd

    # infer what to do with the file
    if ".json" in file:
        try:  # this one will often fail as scores.json is usually formatted differently
            df = pd.read_json(file)
        except ValueError:
            df = parse_scorefile_linear(file)
        # fix df index and return df
        df.set_index(df.index.astype(str).str.replace(find, replace), inplace=True)
        if overwrite:
            df.to_json(file, orient="records")
            return None
        else:
            return df
    elif ".csv" in file:
        df = pd.read_csv(file)
        # fix df index and return df
        df.set_index(df.index.astype(str).str.replace(find, replace), inplace=True)
        if overwrite:
            df.to_json(file, orient="records")
            return None
        else:
            return df
    elif ".list" in file or ".pair" in file:
        # fix and return list or overwrite file
        out_lines = []
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if find in line:
                    line = line.replace(find, replace)
                out_lines.append(line)
        if overwrite:
            with open(file, "w") as f:
                for line in out_lines:
                    print(line, file=f)
            return None
        else:
            return out_lines
    else:  # assume it's a single path, fix and return
        fixed_path = file.replace(find, replace)
        return fixed_path


def df_to_fastas(
    df: pd.DataFrame,
    prefix: str,
    out_path: Optional[str] = None,
    exclude: Optional[str] = None,
) -> pd.DataFrame:
    """
    :param: df: pandas dataframe.
    :param: prefix: prefix for column names to pull sequence from.
    :param: out_path: path to write fasta to.
    :return: None.
    Generate fastas from each row of a df. If `out_path` is not specified, assume the
    index of the dataframe is absolute paths to pdb.bz2 files, a la PyRosettaCluster.
    """
    import sys
    from pathlib import Path

    import pandas as pd
    from tqdm.auto import tqdm

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.mpnn import dict_to_fasta

    tqdm.pandas()

    # get columns that have sequences based on prefix
    sequence_cols = [col for col in df.columns if (prefix in col) and (col != exclude)]

    def mask(row: pd.Series, out_path: Optional[str] = None) -> str:
        name = row.name
        seq_dict = {col: row[col] for col in sequence_cols}
        if out_path is None:  # assume the index of the dataframe is abspaths to pdb.bz2
            # use pathlib sorcery to get the basename
            out_path = f"{Path(name).with_suffix('').with_suffix('')}.fa"
            if "decoys" in out_path:
                out_path = out_path.replace("decoys", "fastas")
            else:
                pass
        else:  # assume the user knows what they're doing :(
            pass
        dict_to_fasta(seq_dict, out_path)

        return out_path

    df["fasta_path"] = df.progress_apply(mask, args=(out_path), axis=1)
    return df


def get_yml() -> str:
    """
    Inspired by pyrosetta.distributed.cluster.converter_tasks.get_yml()
    Works on jupyterhub
    """
    import subprocess
    import sys
    from pathlib import Path

    from pyrosetta.distributed.cluster.config import source_domains

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.utils.io import cmd

    conda_path = f"{sys.prefix}/bin/conda"
    export_command = f"{conda_path} env export --prefix {sys.prefix}"
    try:
        raw_yml = cmd(export_command)

    except subprocess.CalledProcessError:
        raw_yml = ""

    return (
        (
            os.linesep.join(
                [
                    line
                    for line in raw_yml.split(os.linesep)
                    if all(
                        source_domain not in line for source_domain in source_domains
                    )
                    and all(not line.startswith(s) for s in ["name:", "prefix:"])
                    and line
                ]
            )
            + os.linesep
        )
        if raw_yml
        else raw_yml
    )


def parse_scorefile_oneshot(scores: str) -> pd.DataFrame:
    """
    :param: scores: path to scores.json
    :return: pandas dataframe of scores
    Read in a scores.json from PyRosettaCluster in a single shot.
    Memory intensive for a larger scorefile because it does a matrix transposition.
    """
    import pandas as pd

    scores = pd.read_json(scores, orient="records", typ="frame", lines=True)
    scores = scores.T
    mat = scores.values
    n = mat.shape[0]
    dicts = list(mat[range(n), range(n)])
    index = scores.index
    tabulated_scores = pd.DataFrame(dicts, index=index)
    return tabulated_scores


def parse_scorefile_linear(scores: str) -> pd.DataFrame:
    """
    :param: scores: path to scores.json
    :return: pandas dataframe of scores
    Read in a scores.json from PyRosettaCluster line by line.
    Uses less memory thant the oneshot method but takes longer to run.
    """
    import pandas as pd
    from tqdm.auto import tqdm

    dfs = []
    with open(scores, "r") as f:
        for line in tqdm(f.readlines()):
            dfs.append(pd.read_json(line).T)
    tabulated_scores = pd.concat(dfs)
    return tabulated_scores


def pymol_selection(pose: Pose, selector: ResidueSelector, name: str = None) -> str:
    """
    :param: pose: Pose object.
    :param: selector: ResidueSelector object.
    :param: name: name of selection.
    :return: pymol selection string.
    """

    import pyrosetta

    pymol_metric = (
        pyrosetta.rosetta.core.simple_metrics.metrics.SelectedResiduesPyMOLMetric(
            selector
        )
    )
    if name is not None:
        pymol_metric.set_custom_type(name)
    return pymol_metric.calculate(pose)


def print_timestamp(
    print_str: str, start_time: Union[int, float], end: str = "\n", *args
) -> None:
    """
    :param: print_str: string to print
    :param: start_time: start time in seconds
    :param: end: end string
    :param: args: arguments to print_str
    :return: None
    Print a timestamp to the console along with the string passed in.
    """
    from time import time

    time_min = (time() - start_time) / 60
    print(f"{time_min:.2f} min: {print_str}", end=end)
    for arg in args:
        print(arg, end=end)
    return


# Much of the following is extensively borrowed from pyrosetta.distributed.cluster.io


@requires_init
def get_instance_and_metadata(
    kwargs: Dict[Any, Any]
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """
    :param: kwargs: keyword arguments that need to be split into instance and metadata.
    :return: tuple of instance and metadata.
    Get the current state of the PyRosettaCluster instance, and split the
    kwargs into the PyRosettaCluster instance kwargs and ancillary metadata.
    """
    import pyrosetta

    # Deleted a bunch of instance stuff from the original function here.
    # Could add back in if helpful later, particularly if returning this to object-oriented structure.
    instance_kwargs = {}
    # tracking with kwargs instead of class attributes
    instance_kwargs["compressed"] = kwargs.pop("compressed")
    instance_kwargs["decoy_dir_name"] = kwargs.pop("decoy_dir_name")
    instance_kwargs["environment"] = kwargs.pop("environment")
    instance_kwargs["output_path"] = kwargs.pop("output_path")
    instance_kwargs["score_dir_name"] = kwargs.pop("score_dir_name")
    instance_kwargs["simulation_name"] = kwargs.pop("simulation_name")
    instance_kwargs["simulation_records_in_scorefile"] = kwargs.pop(
        "simulation_records_in_scorefile"
    )

    instance_kwargs["tasks"] = kwargs.pop("task")
    for option in ["extra_options", "options"]:
        if option in instance_kwargs["tasks"]:
            instance_kwargs["tasks"][option] = pyrosetta.distributed._normflags(
                instance_kwargs["tasks"][option]
            )
    # the following works if this is called from the same thread as init was called
    instance_kwargs["seeds"] = [pyrosetta.rosetta.numeric.random.rg().get_seed()]

    return instance_kwargs, kwargs


def get_output_dir(base_dir: str) -> str:
    """
    :param: base_dir: base directory to write outputs into.
    :return: output directory with subdirectories auto-generated.
    Get the output directory in which to write files to disk.
    """

    zfill_value = 4
    max_dir_depth = 1000
    try:
        decoy_dir_list = os.listdir(base_dir)
    except FileNotFoundError:
        decoy_dir_list = []
    if not decoy_dir_list:
        new_dir = str(0).zfill(zfill_value)
        output_dir = os.path.join(base_dir, new_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        top_dir = list(reversed(sorted(decoy_dir_list)))[0]
        if len(os.listdir(os.path.join(base_dir, top_dir))) < max_dir_depth:
            output_dir = os.path.join(base_dir, top_dir)
        else:
            new_dir = str(int(top_dir) + 1).zfill(zfill_value)
            output_dir = os.path.join(base_dir, new_dir)
            os.makedirs(output_dir, exist_ok=True)

    return output_dir


def format_result(result: Union[Pose, PackedPose]) -> Tuple[str, Dict[Any, Any]]:
    """
    :param: result: Pose or PackedPose object.
    :return: tuple of (pdb_string, metadata)
    Given a `Pose` or `PackedPose` object, return a tuple containing
    the pdb string and a scores dictionary.
    """

    _pdbstring = io.to_pdbstring(result)
    _scores_dict = io.to_dict(result)
    _scores_dict.pop("pickled_pose", None)

    return (_pdbstring, _scores_dict)


def parse_results(
    results: Union[
        Iterator[Optional[Union[Pose, PackedPose]]],
        Optional[Union[Pose, PackedPose]],
    ]
) -> Union[List[Tuple[str, Dict[Any, Any]]], NoReturn]:
    """
    :param: results: Iterator of Pose or PackedPose objects, which may be None.
    :return: list of tuples of (pdb_string, scores_dict) or nothing if None input.
    Format output results on distributed worker. Input argument `results` can be a
    `Pose` or `PackedPose` object, or a `list` or `tuple` of `Pose` and/or `PackedPose`
    objects, or an empty `list` or `tuple`. Returns a list of tuples, each tuple
    containing the pdb string and a scores dictionary.
    """

    if isinstance(
        results,
        (
            Pose,
            PackedPose,
        ),
    ):
        if not io.to_pose(results).empty():
            out = [format_result(results)]
        else:
            out = []
    elif isinstance(results, collections.abc.Iterable):
        out = []
        for result in results:
            if isinstance(
                result,
                (
                    Pose,
                    PackedPose,
                ),
            ):
                if not io.to_pose(result).empty():
                    out.append(format_result(result))
            else:
                raise OutputError(result)
    elif not results:
        out = []
    else:
        raise OutputError(results)

    return out


def save_results(results: Any, kwargs: Dict[Any, Any]) -> None:
    """
    :param: results: results to pass to `parse_results`
    :param: kwargs: instance and metadata kwargs
    :return: None
    Write results and kwargs to disk after processing metadata. Use `save_results` to
    write results to disk.
    """

    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
    REMARK_FORMAT = "REMARK PyRosettaCluster: "
    compressed = kwargs["compressed"]
    decoy_dir_name = kwargs["decoy_dir_name"]
    environment_file = kwargs["environment"]
    output_path = kwargs["output_path"]
    score_dir_name = kwargs["score_dir_name"]
    simulation_name = kwargs["simulation_name"]
    simulation_records_in_scorefile = kwargs["simulation_records_in_scorefile"]

    # Parse and save results
    for pdbstring, scores in parse_results(results):
        output_dir = get_output_dir(base_dir=os.path.join(output_path, decoy_dir_name))
        decoy_name = "_".join([simulation_name, uuid.uuid4().hex])
        output_file = os.path.join(output_dir, decoy_name + ".pdb")
        if compressed:
            output_file += ".bz2"
        # assume the score_dir is in the same parent dir as the decoy_dir
        # this is a less robust but more thread safe way to get mirrored score files
        score_dir = os.path.join(output_dir.replace(decoy_dir_name, score_dir_name, 1))
        os.makedirs(score_dir, exist_ok=True)
        score_file = os.path.join(score_dir, decoy_name + ".json")
        extra_kwargs = {
            "crispy_shifty_decoy_name": decoy_name,
            "crispy_shifty_output_file": output_file,
        }
        if os.path.exists(environment_file):
            extra_kwargs["crispy_shifty_environment_file"] = environment_file
        if "crispy_shifty_datetime_start" in kwargs:
            datetime_end = datetime.now().strftime(DATETIME_FORMAT)
            duration = str(
                (
                    datetime.strptime(datetime_end, DATETIME_FORMAT)
                    - datetime.strptime(
                        kwargs["crispy_shifty_datetime_start"],
                        DATETIME_FORMAT,
                    )
                ).total_seconds()
            )  # For build-in functions
            extra_kwargs.update(
                {
                    "crispy_shifty_datetime_end": datetime_end,
                    "crispy_shifty_total_seconds": duration,
                }
            )
        instance, metadata = get_instance_and_metadata(
            toolz.dicttoolz.keymap(
                lambda k: k.split("crispy_shifty_")[-1],
                toolz.dicttoolz.merge(extra_kwargs, kwargs),
            )
        )
        pdbfile_data = json.dumps(
            {
                "instance": collections.OrderedDict(sorted(instance.items())),
                "metadata": collections.OrderedDict(sorted(metadata.items())),
                "scores": collections.OrderedDict(sorted(scores.items())),
            }
        )
        # Write full .pdb record
        pdbstring_data = pdbstring + os.linesep + REMARK_FORMAT + pdbfile_data
        if compressed:
            with open(output_file, "wb") as f:
                f.write(bz2.compress(str.encode(pdbstring_data)))
        else:
            with open(output_file, "w") as f:
                f.write(pdbstring_data)
        if simulation_records_in_scorefile:
            scorefile_data = pdbfile_data
        else:
            scorefile_data = json.dumps(
                {
                    metadata["output_file"]: collections.OrderedDict(
                        sorted(scores.items())
                    ),
                }
            )
        # Write data to new scorefile per decoy
        with open(score_file, "w") as f:
            f.write(scorefile_data)


def wrapper_for_array_tasks(func: Callable, args: List[str]) -> None:

    """
    :param: func: function to wrap
    :param: args: list of arguments to apply to func
    :return: None
    This function wraps a distributable pyrosetta function. It is intended to run once
    per a single task on a single thread on a worker. If it is used on a a worker that
    has multiple threads and is wrapping a function that has multithreading support,
    it might or might not still work but some of the resulting metadata could be wrong.
    Additionally, since it inits pyrosetta, pyrosetta should not have been initialized
    on the worker and should not be initialized in the distributed function.
    The use of the `maybe_init` functionality prevents the former from happening but
    not the latter, and if the former happened, it would not be immediately obvious.
    """

    import argparse
    import copy
    import sys

    import pyrosetta

    parser = argparse.ArgumentParser(
        description="Parses arguments passed to the minimal run.py"
    )
    # required task arguments
    parser.add_argument("-pdb_path", type=str, default="", nargs="*", required=True)
    # optional task arguments
    parser.add_argument("-options", type=str, default="", nargs="*", required=False)
    parser.add_argument(
        "-extra_options", type=str, default="", nargs="*", required=False
    )
    parser.add_argument(
        "-extra_kwargs", type=str, default="", nargs="*", required=False
    )
    # arguments tracked by pyrosettacluster. could add some of the others below in save_kwargs
    parser.add_argument("-instance", type=str, default="", nargs="*", required=True)

    args = parser.parse_args(sys.argv[1:])
    print("Design will proceed with the following options:")
    print(args)

    # The options strings are passed without the leading "-" so that argparse doesn't interpret them as arguments. Read them in,
    # assuming they are a list of key-value pairs where odd-indexed elements are keys and even-indexed elements are values. Add
    # in the leading "-" and pass them to pyrosetta.
    pyro_kwargs = {
        "options": " ".join(
            [
                "-" + args.options[i] + " " + args.options[i + 1]
                for i in range(0, len(args.options), 2)
            ]
        ),
        "extra_options": " ".join(
            [
                "-" + args.extra_options[i] + " " + args.extra_options[i + 1]
                for i in range(0, len(args.extra_options), 2)
            ]
        ),
    }
    pyrosetta.distributed.maybe_init(**pyro_kwargs)

    # Get kwargs to pass to the function from the extra kwargs
    func_kwargs = {
        args.extra_kwargs[i]: args.extra_kwargs[i + 1]
        for i in range(0, len(args.extra_kwargs), 2)
    }

    instance_kwargs = {
        args.instance[i]: args.instance[i + 1] for i in range(0, len(args.instance), 2)
    }

    for pdb_path in args.pdb_path:
        # Add the required kwargs
        print("check")
        func_kwargs["pdb_path"] = pdb_path

        datetime_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Run the function
        pposes = func(**func_kwargs)

        # task_kwargs is everything that would be passed in a task in pyrosetta distributed.
        # This isn't a perfect way of figuring out which are which, but it's the best I can do here easily
        # without deviating too far.
        task_kwargs = copy.deepcopy(func_kwargs)
        task_kwargs.update(pyro_kwargs)
        save_kwargs = {
            "compressed": True,
            "decoy_dir_name": "decoys",
            "score_dir_name": "scores",
            "environment": "",
            "task": task_kwargs,
            "output_path": "~",
            "simulation_name": "",
            "simulation_records_in_scorefile": False,
            "crispy_shifty_datetime_start": datetime_start,
        }
        save_kwargs.update(instance_kwargs)
        save_results(pposes, save_kwargs)


def gen_array_tasks(
    distribute_func: str,
    design_list_file: str,
    output_path: str,
    queue: Optional[str] = "cpu",
    extra_kwargs: Optional[dict] = {},
    cores: Optional[int] = None,
    gres: Optional[str] = None,
    memory: Optional[str] = "4G",
    nstruct: Optional[int] = 1,
    nstruct_per_task: Optional[int] = 1,
    options: Optional[str] = "",  # options for pyrosetta initialization
    perlmutter_mode: Optional[bool] = False,
    sha1: Optional[str] = "",
    simulation_name: Optional[str] = "crispy_shifty",
    time: Optional[str] = "",
    func_root: Optional[str] = None,
):
    """
    :param: distribute_func: function to distribute, formatted as a `str`:
    "module.function". The function must be a function that takes a list of pdb paths
    and returns or yields a list of poses.
    :param: design_list_file: path to a file containing a list of pdb paths.
    :param: output_path: path to the directory where the results will be saved.
    :param: queue: name of the queue to submit to on SLURM.
    :param: extra_kwargs: A `dict` object specifying extra kwargs to pass to the
    function being distributed.
    :param: cores: number of cores to use per task.
    :param: gres: string specifying the gres to use, if any, e.g. "--gres=gpu:1".
    :param: memory: amount of memory to request for each task.
    :param: nstruct: number of structures to generate per input.
    :param: nstruct_per_task: number of structures to generate per task generated. Using
    this option will chunk files from the design_list_file together, resulting in less
    total tasks, useful for fast running protocols.
    :param: options: options for pyrosetta initialization.
    :param: perlmutter_mode: whether to use the Perlmutter mode. Perlmutter mode
    grabs entire nodes at a time and uses GNU parallel to run the tasks on the nodes.
    It assumes you will be using GPU nodes. It therefore also does not support the
    cores, gres, memory and queue arguments.
    :param: sha1: A `str` or `NoneType` object specifying the git SHA1 hash string of
    the particular git commit being simulated. If a non-empty `str` object is provided,
    then it is validated to match the SHA1 hash string of the current HEAD,
    and then it is added to the simulation record for accounting. If an empty string
    is provided, then ensure that everything in the working directory is committed
    to the repository. If `None` is provided, then bypass SHA1 hash string
    validation and set this attribute to an empty string. From PyRosettaCluster.
    Defaults to "".
    :param: simulation_name: A `str` or `NoneType` object specifying the name of the
    specific simulation being run.
    :time: `str` specifying walltime. Must be compatible with queue. Example `55:00`
    would be 55 min.
    """
    import os
    import stat
    import sys
    from pathlib import Path

    import git
    from more_itertools import ichunked
    from pyrosetta.distributed.cluster.converters import _parse_sha1 as parse_sha1
    from tqdm.auto import tqdm

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.utils.io import get_yml

    sha1 = parse_sha1(sha1)
    # if the user provided None then sha1 is still an empty string, this fixes that
    if sha1 == "":
        sha1 = "untracked"
    else:
        pass
    # use git to find the root of the repo
    repo = git.Repo(str(Path(__file__).resolve()), search_parent_directories=True)
    root = repo.git.rev_parse("--show-toplevel")
    if func_root is None:
        func_root = root
    env_path = str(Path(root) / "envs" / "shifty")
    if os.path.exists(env_path):
        pass
    else:  # crispy env must be installed in envs/crispy or must be used on DIGS
        env_path = "/projects/crispy_shifty/envs/shifty"
    python = env_path + "/bin/python"

    os.makedirs(output_path, exist_ok=True)

    # Make a task generator that can scale up sampling
    def create_tasks(
        design_list_file, options, nstruct_per_task
    ) -> Iterator[Dict[Any, Any]]:
        """
        :param: design_list_file: path to a file containing a list of input files
        :param: options: options for pyrosetta initialization
        :param: nstruct_per_task: number of structures to generate per task
        :return: an iterator of task dicts.
        Generates tasks for pyrosetta distributed-style array tasks
        TODO: docstring, better type annotations.
        """
        with open(design_list_file, "r") as f:
            # returns an iteratable with nstruct_per_task elements: lines of design_list_file
            for lines in ichunked(f, nstruct_per_task):
                tasks = {"-options": ""}
                tasks["-extra_options"] = options
                # join the lines of design_list_file with spaces, removing trailing newlines
                tasks["-pdb_path"] = " ".join(line.rstrip() for line in lines)
                yield tasks

    jid = "{SLURM_JOB_ID%;*}"
    sid = "{SLURM_ARRAY_TASK_ID}p"

    slurm_dir = os.path.join(output_path, "slurm_logs")
    os.makedirs(slurm_dir, exist_ok=True)

    tasklist = os.path.join(output_path, "tasks.cmds")
    # Save active conda environment as a yml file
    env_file = os.path.join(output_path, "environment.yml")
    env_str = get_yml()
    with open(env_file, "w") as f:
        f.write(env_str)

    # run_sh format is the only difference in perlmutter mode vs non-perlmutter mode
    if perlmutter_mode:  # perlmutter mode ignores cores, gres, memory and queue.
        if time == "":
            time = "55:00"
        else:
            pass
        run_sh = "".join(
            [
                "#!/bin/bash\n",
                "#SBATCH --account=m4129_g\n",  # equivalent to -A
                "#SBATCH --constraint=gpu\n",  # equivalent to -C
                f"#SBATCH --job-name={simulation_name}\n",
                "#SBATCH --gpus=4\n",  # equivalent to -G
                "#SBATCH --nodes=1\n",
                "#SBATCH --ntasks=4\n",
                f"#SBATCH -e {slurm_dir}/{simulation_name}-%A_%a.err \n",
                f"#SBATCH -o {slurm_dir}/{simulation_name}-%A_%a.out \n",
                "#SBATCH -q regular\n",
                f"#SBATCH --time={time}\n",  # the shorter the better within reason
                "module load cudatoolkit/11.5\n",
                "N=$(( SLURM_ARRAY_TASK_ID - 1 ))\n",
                "N2=$(( N + 1 ))\n",
                "start_idx=$(( N*4 + 1 ))\n",
                "end_idx=$(( N2*4 ))\n",
                f"source activate {env_path}\n",
                f"""head -n $end_idx {tasklist} | tail -n +$start_idx | parallel 'CUDA_VISIBLE_DEVICES=$(("{{%}}" - 1)) && bash -c {{}}'""",
            ]
        )
    else:
        if cores is not None:
            cores_string = f"#SBATCH -c {cores}\n"
        else:
            cores_string = "\n"
        if gres is not None:
            gres_string = f"#SBATCH {gres}\n"
        else:
            gres_string = "\n"
        if time == "":
            time_string = "\n"
        else:
            time_string = f"#SBATCH --time={time}\n"

        run_sh = "".join(
            [
                "#!/usr/bin/env bash \n",
                f"#SBATCH -J {simulation_name} \n",
                cores_string,
                f"#SBATCH -e {slurm_dir}/{simulation_name}-%J.err \n",
                f"#SBATCH -o {slurm_dir}/{simulation_name}-%J.out \n",
                f"#SBATCH -p {queue} \n",
                f"#SBATCH --mem={memory} \n",
                gres_string,
                time_string,
                "\n",
                f"JOB_ID=${jid} \n",
                f"""CMD=$(sed -n "${sid}" {tasklist}) \n""",
                f"""echo "${{CMD}}" | bash""",
            ]
        )
    # Write the run.sh file
    run_sh_file = os.path.join(output_path, "run.sh")
    with open(run_sh_file, "w+") as f:
        print(run_sh, file=f)
    # Make the run.sh executable
    st = os.stat(run_sh_file)
    os.chmod(run_sh_file, st.st_mode | stat.S_IEXEC)

    func_split = distribute_func.split(".")
    func_name = func_split[-1]
    path_inserts = [
        # use the root of the repo location to import the module
        f"sys.path.insert(0, '{root}')\n",
    ]
    if func_root != root:
        path_inserts.append(f"sys.path.insert(0, '{func_root}')\n")
    lines = [
        f"#!{python} \n",
        "import sys\n",
        "from crispy_shifty.utils.io import wrapper_for_array_tasks\n",
        f"from {'.'.join(func_split[:-1])} import {func_name}\n",
        f"wrapper_for_array_tasks({func_name}, sys.argv)",
    ]
    lines[2:2] = path_inserts
    run_py = "".join(lines)
    run_py_file = os.path.join(output_path, "run.py")
    # Write the run.py file
    with open(run_py_file, "w+") as f:
        print(run_py, file=f)
    # Make the run.py executable
    st = os.stat(run_py_file)
    os.chmod(run_py_file, st.st_mode | stat.S_IEXEC)

    instance_dict = {
        "output_path": output_path,
        "simulation_name": simulation_name,
        "environment": env_file,
        "sha1": sha1,
    }

    instance_str = "-instance " + " ".join(
        [" ".join([k, str(v)]) for k, v in instance_dict.items()]
    )
    extra_kwargs_str = "-extra_kwargs " + " ".join(
        [" ".join([k, str(v)]) for k, v in extra_kwargs.items()]
    )

    # Track the number of tasks in the tasklist
    count = 0
    with open(tasklist, "w+") as f:
        for i in range(0, nstruct):
            for tasks in create_tasks(design_list_file, options, nstruct_per_task):
                task_str = " ".join([" ".join([k, str(v)]) for k, v in tasks.items()])
                cmd = f"{run_py_file} {task_str} {extra_kwargs_str} {instance_str}"
                print(cmd, file=f)
                count += 1

    # the number of tasks for the array depends on whether it is perlmutter mode or not
    if not perlmutter_mode:
        array_len = count
    else:  # in perlmutter mode we run tasks in chunks of 4, so we need to divide by 4
        if count % 4 == 0:
            array_len = count / 4
        else:  # and if there are any left over, we need to add 1 to the array length
            array_len = (count // 4) + 1

    # Let's go
    print("Run the following command with your desired environment active:")
    print(f"sbatch -a 1-{int(array_len)} {run_sh_file}")


def collect_score_file(output_path: str, score_dir_name: str = "scores") -> None:
    """
    :param: output_path: path to the directory where the score dir is in.
    :param: score_dir_name: name of the directory where the score files are in.
    :return: None
    Collects all the score files in the `score_dir_name` subdirectory of the
    `output_path` directory. Concatenates them into a single file in the
    `output_path` directory.
    """

    import os
    from glob import iglob

    score_dir = os.path.join(output_path, score_dir_name)
    with open(os.path.join(output_path, "scores.json"), "w") as scores_file:
        for score_file in iglob(os.path.join(score_dir, "*", "*.json")):
            with open(score_file, "r") as f:
                scores_file.write(f.read() + "\n")
    return


def collect_and_clean_score_file(
    output_path: str, drop: str, score_dir_name: str = "scores"
) -> None:
    """
    :param: output_path: path to the directory where the score dir is in.
    :param: drop: keys containing this string will be erased from scorefiles.
    :param: score_dir_name: name of the directory where the score files are in.
    :return: None
    Collects all the score files in the `score_dir_name` subdirectory of the
    `output_path` directory. Concatenates them into a single file in the
    `output_path` directory.
    """

    import json
    import os
    from glob import iglob

    from tqdm.auto import tqdm

    score_dir = os.path.join(output_path, score_dir_name)
    with open(os.path.join(output_path, "scores.json"), "w") as scores_file:
        for score_file in tqdm(iglob(os.path.join(score_dir, "*", "*.json"))):
            with open(score_file, "r") as f:
                full_score_dict = json.loads(f.read())
                cleaned_score_dict = {}
                for index_key, score_dict in full_score_dict.items():
                    cleaned_score_dict[index_key] = {
                        k: v for k, v in score_dict.items() if drop not in k
                    }
            with open(score_file, "w") as f:
                print(json.dumps(cleaned_score_dict), file=f)
            scores_file.write(json.dumps(cleaned_score_dict) + "\n")
    return


@requires_init
def test_func(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:

    import sys
    from pathlib import Path

    import pycorn  # test env
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
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    for pose in poses:
        print("hi!")

        ppose = io.to_packed(pose)
        yield ppose

def renumber_pdb_from(pdb,n=1):
    #renumber pdbs much faster than using rosetta
    from biopandas.pdb import PandasPdb
    pdbtype=pdb.split('.')[-1]
    name=pdb.replace('.pdb','').replace('.gz','').replace('.bz2','')
    if pdbtype=="bz2":
        import bz2
        with bz2.open(pdb,'rt') as bf:
            with open(f"{name}_renumber_tmp.pdb",'w') as pf:
                while line:= bf.readline():
                    pf.write(line)
        pdb=f"{name}_renumber_tmp.pdb"
    pdb_df = PandasPdb().read_pdb(pdb)
    numdict={k:v+n for v,k in enumerate(pdb_df.df["ATOM"]["residue_number"].unique())}
    pdb_df.df["ATOM"]["residue_number"]=[numdict[x] for x in pdb_df.df["ATOM"]["residue_number"]]
    if pdbtype=="pdb":
        pdb_df.to_pdb(path=f'{name}_from_{n}.pdb', 
                records=['ATOM', 'HETATM', 'ANISOU', 'OTHERS'], 
                gz=False, 
                append_newline=True)
    elif pdbtype=="gz":
        pdb_df.to_pdb(path=f'{name}_from_{n}.pdb', 
                records=['ATOM', 'HETATM', 'ANISOU', 'OTHERS'], 
                gz=True, 
                append_newline=True)
    elif pdbtype=="bz2":
        pdb_df.to_pdb(path=f'{name}_renumber_tmp_out.pdb', 
                records=['ATOM', 'HETATM', 'ANISOU', 'OTHERS'], 
                gz=False, 
                append_newline=True)
        with bz2.open(f'{name}_from_{n}.pdb.bz2','wt') as bf:
            with open(f"{name}_renumber_tmp_out.pdb",'r') as pf:
                while line:= pf.readline():
                    bf.write(line)
            import os
            os.remove(f"{name}_renumber_tmp_out.pdb")
            os.remove(f"{name}_renumber_tmp.pdb")
