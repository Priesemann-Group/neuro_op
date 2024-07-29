#!/bin/bash

# Use bash as shell
#$ -S /bin/bash

rundir=/data.nst/jfriedel/projects/neuro_op_data/@n1_amor/2024-07C1

# Preserve environment variables
#$ -V

# Execute from current working directory
#$ -cwd

# Merge standard output and standard error into one file
#$ -j yes

# Standard name of the job (if none is given on the command line)
#$ -N demonstrationParallel

# Some diagnostic messages for the output
echo Started: `date`
echo          on `hostname`
echo ------------

# source /core/uge/LMP/common/settings.sh
# qsub -q rostam.q -t 1:20 submissionscript.sh

# Check number of input arguments
if [ $# -ne 0 ]
then
  echo "Usage: parallel_job"
  exit 65
fi

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


# Print the command used (e.g., for later error analysis)
# Make use of environment variables set by SGE
echo Number of slots: $NSLOTS, ${SGE_TASK_ID}

source /usr/ds/anaconda3-2022.05/bin/activate /data.nst/jfriedel/envs/neuro_op/

mkdir -p $rundir
cd $rundir
# Execute the above commands
python ./computation.py ${SGE_TASK_ID} &
# Wait for all the above processes to exit
wait

mkdir -p input output
mv in*.pkl input/
mv out*.h5 output/ 

# Print diagnostic messages
echo ------------
echo Ended: `date`
