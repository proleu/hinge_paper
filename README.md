### Common modules, methods, files and scripts for Praetorius and Leung et al. 2023

Included environments are tested on Ubuntu 20.04
The majority of notebooks and many util scripts are intended to be run on computational resources used for this work, such as our own HPC cluster and NERSC perlmutter. It is likely possible to modify the scripts to work on any cluster with a SLURM scheduler, and intended to be straightforward to do so.

The design pipeline described in this work consists of the notebooks described below run in order. Input scaffolds for the first notebook are from references(22, 23) and are available upon reasonable request. All subsequent notebooks take a list of input designs, and map them through a design or analysis function, then gather and analyze the data and select a subset to make a list for the next notebook. Generally, the function name and path in the codebase can be inferred from the `distribute_func` parameter in the notebook cell that runs `gen_array_tasks`.

The following are jupyter notebooks in the order they were run and a short description.

Notebooks 0-2 in `./notebooks`
* 00_filter_scaffold_sets.ipynb: Selects subsets of input scaffolds based on computational metrics.
* 01_prep_inputs.ipynb:  Collects additional metadata and "domesticates" the scaffolds  by removing disulfides and terminal unstructured regions. 
* 02_make_states.ipynb: Generates the alternative backbone conformations for the hinges  using the alignment based docking approach shown in Figure 1.

Notebooks 0-8 in `./projects/crispy_shifties`
* 00_design_bound_states.ipynb: One-state design of the alternative conformations generated in the previous notebook. 
* 01_loop_bound_states.ipynb: Loop closure between the hinge domains to make state Y.
* 02_mpnn_bound_states.ipynb: One state MPNN design of state Y -peptide complexes.
* 03_fold_bound_states.ipynb: AF2-IG of the complexes generated in the previous step.
* 04_pair_bound_states.ipynb: Relooping of parent scaffolds to match exactly the secondary structure of the hinge in state Y, generating state X and Y pair.
* 05_design_paired_states.ipynb: Rosetta MSD of the paired state X and state Y.
* 06_mpnn_paired_states.ipynb: MPNN-MSD of the paired state X and state Y.
* 07_fold_paired_states_Y.ipynb: AF2-IG of the designs in state Y.
* 08_fold_paired_states_X.ipynb: AF2 of the designs in state X.
* 09_filter_and_order.ipynb: Rosetta filtering and ordering as performed for cs200-cs223.
* 09_resurface_filter.ipynb: Rosetta filtering and effector peptide resurfacing as performed for cs224-295.
* 10_analyze_and_order.ipynb: Filtering and ordering as performed for cs224-cs295.
* 11_make_extensions.ipynb: Generate extended versions of hinges for FRET constructs. 

Notebooks 0-4 in `./projects/DAB` were run in order to make 3hbs.
* 01_inpaint_structure.ipynb: Add additional helices to peptide effectors.
* 02_mpnn_inpaints.ipynb: MPNN design of the 3hb effector bound to state Y hinge. 
* 03_fold_complex.ipynb: AF2-IG of the hinge + 3hb effector complex. 
* 04_fold_monomer.ipynb: AF2 of the 3hbs alone. 
