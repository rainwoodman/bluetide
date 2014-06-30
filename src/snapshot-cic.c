#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <bigfile-mpi.h>
#include "cic.h"
int ThisTask;
int NTask;

static void cicadd(CIC * cic, BigFile * bf, char * blockname, char * mblockname) {

    BigBlock bb = {0};
    BigBlock mbb = {0};

    if(0 != big_file_mpi_open_block(bf, &bb, blockname, MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open %s: %s\n", blockname, big_file_get_error_message()); 
        exit(1);
    } 
    if(0 != big_file_mpi_open_block(bf, &mbb, mblockname, MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open %s: %s\n", mblockname, big_file_get_error_message()); 
        exit(1);
    } 
    BigArray array = {0};
    BigArray marray = {0};
    ptrdiff_t offset = ThisTask * bb.size / NTask;
    ptrdiff_t end = (ThisTask + 1) * bb.size / NTask;
    size_t chunksize = 1024 * 1024 * 128;
    while(offset < end) {
        if(0 != big_block_read_simple(&bb, offset, chunksize, &array, "f8")) {
            fprintf(stderr, "failed to read: %s\n", blockname, big_file_get_error_message());
            exit(1);
        }
        if(0 != big_block_read_simple(&mbb, offset, chunksize, &marray, "f4")) {
            fprintf(stderr, "failed to read: %s\n", mblockname, big_file_get_error_message());
            exit(1);
        }

        ptrdiff_t i;
        double * pos = array.data;
        double * mass = marray.data;
#pragma omp parallel for 
        for(i = 0; i < array.dims[0]; i ++) {
            cic_add_particle(cic, &pos[3 * i], mass[i]);
        }
        free(array.data); 
        free(marray.data); 
        offset += array.dims[0];
    }
    big_block_mpi_close(&bb, MPI_COMM_WORLD);
}

int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    BigFile bf = {0};

    BigBlock bb = {0};
    CIC cic = {0};
    double BoxSize;
    int Ngrid = atoi(argv[2]); 
    if(0 != big_file_mpi_open(&bf, argv[1], MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open %s: %s\n", argv[1], big_file_get_error_message()); 
        exit(1);
    } 
    if(0 != big_file_mpi_open_block(&bf, &bb, "header", MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open %s: %s\n", argv[1], big_file_get_error_message()); 
        exit(1);
    }
    
    big_block_get_attr(&bb, "BoxSize", &BoxSize, "f8", 1);

    big_block_mpi_close(&bb, MPI_COMM_WORLD);

    fprintf(stderr, "BoxSize = %g\n", BoxSize);
    cic_init(&cic, Ngrid, BoxSize);
    cicadd(&cic, &bf, "0/Position", "0/Mass");
    cicadd(&cic, &bf, "1/Position", "1/Mass");
    cicadd(&cic, &bf, "4/Position", "4/Mass");
    cicadd(&cic, &bf, "5/Position", "5/Mass");

    big_file_mpi_close(&bf, MPI_COMM_WORLD);
    double * reduced = malloc(sizeof(double) * Ngrid * Ngrid * Ngrid);

    MPI_Reduce(cic.buffer, reduced, Ngrid * Ngrid * Ngrid, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {
        fwrite(reduced, sizeof(double), Ngrid * Ngrid * Ngrid, stdout);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}

