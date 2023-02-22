OUTPUT_FOLDER = bin

open-mpi-all: open-mpi-build open-mpi-run

open-mpi-build:
	mpicc src/open-mpi/mpi.c -o ${OUTPUT_FOLDER}/mpi -lm

open-mpi-run:
	mpirun -n 4 ${OUTPUT_FOLDER}/mpi