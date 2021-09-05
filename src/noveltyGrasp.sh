#!/bin/bash
#PBS -q beta
#PBS -l select=1:ncpus=24
#PBS -l walltime=3:00:00
#PBS -N noveltyGrasp
#PBS -J 1-10

# this script must be used with qsub on MeSU beta
# python novelty search algorithm for grasping
# !!! Change futures.map to Pool().map in noveltysearch.py otherwise the program never finishes !!!

echo "PBS_O_WORKDIR = $PBS_O_WORKDIR"
echo "USER = $USER"
echo "HOST = $HOST"
echo "PBS_JOBID = $PBS_JOBID"
echo "PBS_JOBNAME = $PBS_JOBNAME"
echo "PBS_NODEFILE = $PBS_NODEFILE"
echo "PBS_NUM_PPN = $PBS_NUM_PPN"
echo "PBS_NCPUS = $PBS_NCPUS"

# copy scripts to the current temporary directory
GRABPATH=/scratchbeta/$USER/grab-babling/src
if [ ! -d "$GRABPATH" ]; then
	echo "directory ${GRABPATH} does not exist"
	exit 1
fi
cp $GRABPATH/applynoveltygrasping.py $GRABPATH/controllers.py $GRABPATH/noveltysearch.py $GRABPATH/utils.py $GRABPATH/plotting.py $GRABPATH/DynamicMovementPrimitives.py .
#cp /scratchbeta/$USER/AurelienMorel/testdeap.py .

mkdir runs

ENVPATH=/home/$USER/venv # python virtual environment
if [ ! -d "$ENVPATH" ]; then
    echo "virtual environment ${ENVPATH} does not exist"
    exit 1
fi
source $ENVPATH/bin/activate
echo `which python`

ROBOT="${ROBOT:-baxter}"
OBJECT="${OBJECT:-sphere}"
FLAG="${FLAG:-}"
REPEAT="${REPEAT:-1}"
POPULATION="${POPULATION:-100}"
GENERATION="${GENERATION:-1000}"
CELLS="${CELLS:-1000}"
MODE="${MODE:-joint positions}"
ALGO="${ALGO:-nsmbs}"
BD="${BD:-pos_div_pos_grip}"
if ["${MODE}" = "pd stable"]; then
    export OPENBLAS_NUM_THREADS=1
fi

python applynoveltygrasping.py -r $ROBOT -o $OBJECT $FLAG -n $REPEAT -p $POPULATION -g $GENERATION -c $CELLS -m "${MODE}" -a $ALGO -d $BD
deactivate

FOLDER="${FOLDER:-runMeSU}"
STOREPATH=/scratchbeta/$USER/${FOLDER} # directory in which results will be stored
DATE=`date +"%Y-%m-%d_%T"`
mkdir -p $STOREPATH # make sure it exists
for run in runs/*; do
    mv "$run" "$STOREPATH/${OBJECT}_${FLAG}_${PBS_JOBID}_${DATE}_${run##*/}_${PBS_JOBNAME}"
done
#mv stdout.txt stderr.txt $STOREFOLDER
