import sys
from pathlib import Path

# insert the root of the repo into the sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crispy_shifty.utils.io import cmd, cmd_no_stderr

tag_256 = "AAATCCCAAAGCCATACCCTAACCTTAATTCAATGCAACTCATTTATATCTATGTAAGTCAGACACATGGAAACGCACTTTTGAACAGTTAGATAGCTACTGATTGCCCCGAAGGCAGGTACGTAGGGACCGTTCTCTGTCCTCCGGAGAGTGAGCGATCGACGGTTGGCTTCGCTGGGTGTGCGTCGGCGCCGCGGGGCCTGCTCGTGGTCTTGTTTCCACCAGCATCACGAGGATGACTAGTATTACAAGAATA"
tag_64 = "AATCCAAGCATTACCTATGAACTTTGCCCGACGTAGTCTCGCGGAGGTGGGCTGTTCACAGATA"


def capture_1shot_domesticator(stdout: str) -> str:
    """split input into lines.
    loop once through discarding lines up to ones including >.
    return joined output"""
    sequence = []
    append = False
    for line in stdout.splitlines():
        if append:
            sequence.append(line)
        else:
            pass
        if ">unknown_seq1" in line:
            append = True
        else:
            pass
    to_return = "".join(sequence)
    return to_return
