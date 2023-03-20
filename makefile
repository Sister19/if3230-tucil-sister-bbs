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

open-mp-all: open-mp-build open-mp-run

open-mp-all-test: open-mp-build open-mp-run-test

open-mp-build:
	gcc src/open-mp/mp.c --openmp -o ${OUTPUT_FOLDER}/mp -lm

open-mp-run:
	./${OUTPUT_FOLDER}/mp

open-mp-run-test:
	./${OUTPUT_FOLDER}/mp < test_case/${tc}.txt

cuda-all: cuda-build cuda-run

cuda-build:
	nvcc ./src/cuda/cuda.cu -o ./${OUTPUT_FOLDER}/cuda -diag-suppress 2464 -lm

cuda-run: 
	./${OUTPUT_FOLDER}/cuda < test_case/${tc}.txt