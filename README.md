### Common modules, methods, files and scripts for Praetorius and Leung et al. 2023

Included environments are tested on Ubuntu 20.04
The majority of notebooks and many util scripts are intended to be run on
computational resources that Institute for Protein Design labs have access to such as the digs, and NERSC perlmutter
It is likely possible to modify the scripts to work on any cluster with a SLURM 
scheduler, and intended to be straightforward to do so.  

For the paper, the final version of the pipeline was as follows:
notebooks 0-2 were run in `./notebooks`
notebook 0 selects input scaffolds, these are available upon reasonable request and are from Brunette et al. 2015 and Brunette et al. 2020. 
(From here on, almost all notebooks take a list of input scaffolds, and map them through a design or analysis function, then gather and analyze the data and select a subset to make a list for the next notebook. 
The function name and path in the codebase can be inferred from `distribute_func` parameter in the notebook cell that runs `gen_array_tasks`
notebook 1 does collects additional metadata and "domesticates" the scaffolds (removing trailing loops and disulfides)
notebook 2 does the backbone design for the hinges (alignment based docking from figure 1).
Then, notebooks 0-8 are run in order in `./projects/crispy_shifties` . 
After this, for cs200-223 notebook `09_filter_and_order` was run to generate that order. 
For cs224-295, notebooks `09_resurface_filter`, and notebook 10 was run to generate that order.
for the 3hbs, notebooks in `./projects/DAB` were run in order in a similar fashion. 
