#!/bin/sh

# loop through all folders in a directory and start a SLURM PIV computration for them 
for FOLDER in $1*/; do
echo ${FOLDER}
 sbatch batch_mpi_piv.sbatch ${FOLDER}
 sleep 1
done
