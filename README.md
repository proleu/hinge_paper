```
           _                       _     _  __ _         
          (_)                     | |   (_)/ _| |        
  ___ _ __ _ ___ _ __  _   _   ___| |__  _| |_| |_ _   _ 
 / __| '__| / __| '_ \| | | | / __| '_ \| |  _| __| | | |
| (__| |  | \__ \ |_) | |_| | \__ \ | | | | | | |_| |_| |
 \___|_|  |_|___/ .__/ \__, | |___/_| |_|_|_|  \__|\__, |
                | |     __/ |                       __/ |
                |_|    |___/                       |___/ 
```


### Common modules, methods, files and scripts for Crispy Shifty efforts
Can be used by running the following:
```
import sys
sys.path.insert(0, "/projects/crispy_shifty/") 
# now can import functions and objects, like:
from crispy_shifty.protocols.states import grow_terminal_helices
...
```
Included environments are tested on Ubuntu 20.04
The majority of notebooks and many util scripts are intended to be run on
computational resources that Institute for Protein Design labs have access to.
It is likely possible to modify the scripts to work on any cluster with a SLURM 
scheduler, and maybe straightforward to do so.  
TODO
