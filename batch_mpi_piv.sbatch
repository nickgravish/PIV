#!/bin/sh
# runs PIV code on the folder of images passed into the code 

#SBATCH -n 128 # Number of cores
#SBATCH -t 20 # Runtime in minutes
#SBATCH -p general # Partition to submit to
#SBATCH --mem-per-cpu=2000 # Memory per cpu in MB (see also --mem)

echo `basename $1`

mpirun -np 128 --mca orte_base_help_aggregate 0 python PIV_MPI_cluster.py \
		$1 \
		> /n/home03/ngravish/logs/`basename $1`_{$SLURM_JOBID}_out.txt 2> /n/home03/ngravish/logs/`basename $1`_{$SLURM_JOBID}_err.txt
