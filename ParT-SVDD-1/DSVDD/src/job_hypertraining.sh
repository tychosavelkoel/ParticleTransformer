cd /data/alice/tqwsavelkoel/ParT-SVDD-1/DSVDD/src

#export PATH=$PBS_O_PATH
source /data/alice/tqwsavelkoel/env/etc/profile.d/conda.sh
conda activate  /data/alice/wvolgering/my_venv_new/


#random_id = $RANDOM

#pip freeze > logfiles/req_${random_id}.txt
python3 hypertuning.py

