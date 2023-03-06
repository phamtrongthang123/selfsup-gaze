#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate env_torchism/
sphinx-apidoc -o docs .
cd docs 
make html