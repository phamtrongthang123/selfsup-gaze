#!/bin/sh 
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
rm -rf env_torchism/
conda create --prefix env_torchism/ python=3.7 -y
conda activate env_torchism/

pip install -r requirements.txt

