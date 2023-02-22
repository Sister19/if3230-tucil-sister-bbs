OUTPUT_FOLDER = bin

all: serial parallel

open-mpi:
# Local
	mpicc src/open-mpi/mpi.c -o $(OUTPUT_FOLDER)/open-mpi -lm

parallel:
# TODO : Parallel compilation
	make open-mpi

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial -lm