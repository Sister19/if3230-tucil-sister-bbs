OUTPUT_FOLDER = bin
n = 4
tc= 32

open-mpi-all: open-mpi-build open-mpi-run

open-mpi-all-test: open-mpi-build open-mpi-run-test

open-mpi-build:
	mpicc src/open-mpi/mpi.c -o ${OUTPUT_FOLDER}/mpi -lm

open-mpi-run:
	mpirun -n ${n} ${OUTPUT_FOLDER}/mpi

open-mpi-run-test:
	mpirun -n ${n} ${OUTPUT_FOLDER}/mpi < test_case/${tc}.txt