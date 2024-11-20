#!/bin/bash
#PBS -N octagon
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4:gpus=1,mem=8000mb,nice=10,walltime=22:00:00
#PBS -m a
#PBS -M velikanm@informatik.uni-freiburg.de
#PBS -j oe
nvidia-smi
echo $(pwd)
cd octagon
. fake_bash.sh
conda activate pfp_env
python -c "import torch; print(torch.cuda.is_available())"
export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/RAFT/core
cd PointFlowMatch
python scripts/train.py log_wandb=True dataloader.num_workers=2 task_name=unplug_charger +experiment=pointflowmatch_so3
echo "Hello World"
exit 0
