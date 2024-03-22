#!/bin/bash
# -----------------------------------------------
#         _           _       _
#        | |__   __ _| |_ ___| |__
#        | '_ \ / _` | __/ __| '_ \
#        | |_) | (_| | || (__| | | |
#        |_.__/ \__,_|\__\___|_| |_|
#                              Fidle at IDRIS
# -----------------------------------------------
#
# SLURM batch script
# Bash script for SLURM batch submission of ci notebooks 
# by Jean-Luc Parouty (CNRS/SIMaP)
#
# Soumission :  sbatch  /(...)/batch_slurm.sh
# Suivi      :  squeue -u $USER

# ==== Job parameters ==============================================

#SBATCH --job-name="Fidle ci"                      # nom du job
#SBATCH --ntasks=1                                 # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                               # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=10                         # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --hint=nomultithread                       # on réserve des coeurs physiques et non logiques
#SBATCH --time=05:00:00                            # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output="FIDLE_CI_%j.out"   		       # nom du fichier de sortie
#SBATCH --error="FIDLE_CI_%j.err"    		       # nom du fichier des erreurs
#SBATCH --mail-user=Someone@somewhere.fr
#SBATCH --mail-type=END,FAIL

# ==== Parameters ==================================================

MODULE_ENV="pytorch-gpu/py3/2.1.1"
RUN_DIR="$WORK/fidle-project/fidle"
CAMPAIN_PROFILE="./fidle/ci/gpu-scale1.yml"
FILTER=( '.*' )

# ==================================================================

echo '------------------------------------------------------------'
echo "Start : $0"
echo '------------------------------------------------------------'
echo "Job id           : $SLURM_JOB_ID"
echo "Job name         : $SLURM_JOB_NAME"
echo "Job node list    : $SLURM_JOB_NODELIST"
echo '------------------------------------------------------------'
echo "module loaded    : $MODULE_ENV"
echo "run dir          : $RUN_DIR"
echo "campain profile  : $CAMPAIN_PROFILE"
echo "filter           : ${FILTER[@]}"
echo '------------------------------------------------------------'

# ---- Module + env.

module purge
module load "$MODULE_ENV"

export PYTHONUSERBASE=$WORK/local/fidle-k3
export PATH=$PATH:$PYTHONUSERBASE/bin

# ---- Run it...

cd "$RUN_DIR"

fid run_ci --quiet --campain "$CAMPAIN_PROFILE" --filter ${FILTER[@]}

echo 'Done.'
