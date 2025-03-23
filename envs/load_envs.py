"""
The file's role is to load a list of environments (i.e. presentation chains) according to need.
functions:
load_all_presentations: load all the presentations from the data file
"""


from importlib import resources
from ast import literal_eval
from envs.ac_moves import simplify_relators


def load_all_presentations():
    file_name = "all_presentations.txt"
    with resources.open_text("data", file_name) as file:
        all_presentations = [literal_eval(line.strip()) for line in file]

    print(f"Loaded {len(all_presentations)} presentations from {file_name}.")

    #because the presentations are not necessarily in simplified form, simplify them before returning

    all_presentations = [simplify_relators(state) for state in all_presentations]
    return all_presentations