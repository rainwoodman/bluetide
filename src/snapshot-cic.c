#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <unistd.h>

#include <bigfile-mpi.h>

#include "cic.h"
int ThisTask;
int NTask;
int ViewSize[3];
double Offset[3] = {0};
int Ngrid;
double BoxSize;

static void cicadd(CIC * cic, CIC * vcic, BigFile * bf, 
        char * blockname, char * mblockname, char * vblockname) {

    BigBlock bb = {0};
    BigBlock mbb = {0};
    BigBlock vbb = {0};

    if(0 != big_file_mpi_open_block(bf, &bb, blockname, MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open %s: %s\n", blockname, big_file_get_error_message()); 
        return;
    } 
    if(0 != big_file_mpi_open_block(bf, &mbb, mblockname, MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open %s: %s\n", mblockname, big_file_get_error_message()); 
        exit(1);
    } 
    if(vcic != NULL) {
        if(0 != big_file_mpi_open_block(bf, &vbb, vblockname, MPI_COMM_WORLD)) {
            fprintf(stderr, "failed to open %s: %s\n", vblockname, big_file_get_error_message()); 
            exit(1);
        } 
    }
    BigArray array = {0};
    BigArray marray = {0};
    BigArray varray = {0};
    ptrdiff_t offset = ThisTask * bb.size / NTask;
    ptrdiff_t end = (ThisTask + 1) * bb.size / NTask;
    size_t chunksize = 1024 * 1024 * 64;
    while(offset < end) {
        size_t readsize = chunksize;
        if(readsize + offset > end) {
            readsize = end - offset;
        }
        if(0 != big_block_read_simple(&bb, offset, readsize, &array, "f8")) {
            fprintf(stderr, "failed to read: %s: %s\n", blockname, big_file_get_error_message());
            exit(1);
        }
        if(0 != big_block_read_simple(&mbb, offset, readsize, &marray, "f4")) {
            fprintf(stderr, "failed to read: %s: %s\n", mblockname, big_file_get_error_message());
            exit(1);
        }
        if(vcic != NULL) {
            if(0 != big_block_read_simple(&vbb, offset, readsize, &varray, "f4")) {
                fprintf(stderr, "failed to read: %s: %s\n", vblockname, big_file_get_error_message());
                exit(1);
            }
        }
        ptrdiff_t i;
        double * pos = array.data;
        float * mass = marray.data;
        float * value = varray.data;
#pragma omp parallel for 
        for(i = 0; i < array.dims[0]; i ++) {
            int d;
            for(d = 0; d < 3; d ++) {
                pos[3 * i + d] -= Offset[d];
            }
            cic_add_particle(cic, &pos[3 * i], mass[i]);
            if(vcic != NULL) {
                cic_add_particle(vcic, &pos[3 * i], mass[i] * value[i]);
            }
        }
        free(array.data); 
        free(marray.data); 
        if(vcic != NULL) {
            free(varray.data); 
        }
        offset += array.dims[0];
        fprintf(stderr, "ThisTask %d move to %td / %td\n", ThisTask, offset, end);
    }
    big_block_mpi_close(&bb, MPI_COMM_WORLD);
    big_block_mpi_close(&mbb, MPI_COMM_WORLD);
    if(vcic != NULL) {
        big_block_mpi_close(&vbb, MPI_COMM_WORLD);
    }   
}
void cicreduce(CIC * cic, CIC * result) {
    MPI_Reduce(cic->buffer, result->buffer, cic->size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void dogas(BigFile * bf, char * Field, char * filename) {
    CIC cic = {0};
    CIC vcic = {0};
    CIC cicreduced = {0};

    cic_init(&cic, BoxSize, Ngrid, ViewSize);
    cic_init(&vcic, BoxSize, Ngrid, ViewSize);
    cic_init(&cicreduced, BoxSize, Ngrid, ViewSize);
    char * vblock = alloca(strlen(Field) + 100);
    sprintf(vblock, "0/%s", Field);
    cicadd(&cic, &vcic, bf, "0/Position", "0/Mass", vblock);
    cicreduce(&cic, &cicreduced);
    memcpy(cic.buffer, cicreduced.buffer, sizeof(double) * cic.size);
    cicreduce(&vcic, &cicreduced);
    memcpy(vcic.buffer, cicreduced.buffer, sizeof(double) * cic.size);
    if(ThisTask == 0) {
        fprintf(stderr, "writing\n");
        FILE * fp = fopen(filename, "w");
        fwrite(cic.buffer, sizeof(double), cic.size, fp);
        fwrite(vcic.buffer, sizeof(double), cic.size, fp);
        fclose(fp);
    }
    cic_destroy(&vcic);
    cic_destroy(&cic);
    cic_destroy(&cicreduced);
}

void domatter(BigFile * bf, char * filename) {
    CIC cic = {0};
    CIC cicreduced = {0};

    cic_init(&cic, BoxSize, Ngrid, ViewSize);
    cic_init(&cicreduced, BoxSize, Ngrid, ViewSize);

    cicadd(&cic, NULL, bf, "0/Position", "0/Mass", NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    if(ThisTask == 0) {
        fprintf(stderr, "0 done\n");
    }
    cicadd(&cic, NULL, bf, "1/Position", "1/Mass", NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    if(ThisTask == 0) {
        fprintf(stderr, "1 done\n");
    }
    cicadd(&cic, NULL, bf, "4/Position", "4/Mass", NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    if(ThisTask == 0) {
        fprintf(stderr, "4 done\n");
    }
    cicadd(&cic, NULL, bf, "5/Position", "5/Mass", NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    if(ThisTask == 0) {
        fprintf(stderr, "5 done\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(ThisTask == 0) {
        fprintf(stderr, "file closed \n");
    }
    cicreduce(&cic, &cicreduced);
    if(ThisTask == 0) {
        fprintf(stderr, "reduce\n");
    }
    if(ThisTask == 0) {
        fprintf(stderr, "writing\n");
        FILE * fp = fopen(filename, "w");
        fwrite(cicreduced.buffer, sizeof(double), cic.size, fp);
        fclose(fp);
    }
    cic_destroy(&cic);
    cic_destroy(&cicreduced);
}

void parse_double3(char * str, double value[3]) {
    char * brk;
    int d = 0;
    for(brk = strtok(str, ","); 
        brk;
        brk = strtok(NULL, ",")) {
        value[d] = atof(brk);
        d ++;
    }
    int last = d - 1;
    for(; d < 3; d++) {
        value[d] = value[last];
    }    
}
void parse_int3(char * str, int value[3]) {
    char * brk;
    int d = 0;
    for(brk = strtok(str, ","); 
        brk;
        brk = strtok(NULL, ",")) {
        value[d] = atoi(brk);
        d ++;
    }
    int last = d - 1;
    for(; d < 3; d++) {
        value[d] = value[last];
    }    
}
void usage() {
    fprintf(stderr, "usage: [-b baryonfield] [-m] [-o offsetx,offsety,offsetz|offsetxyz] [-s sizex,sizey,sizez|sizexyz] filename Ngrid outputprefix\n");
    fprintf(stderr, "-b: include a baryon field\n");
    fprintf(stderr, "-m: do overall mass field\n");
    fprintf(stderr, "-o: offset of view box; code units\n");
    fprintf(stderr, "-s: size of view box; code units\n");
    fprintf(stderr, "Ngrid: number of grids of the full box (resolution is BoxSize / Ngrid, final filesize depend on the viewbox size (-s)) \n");
    exit(1);
}
int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    int opt;
    char * GasField = NULL;
    int DoMatter = 0;
    double Size[3] = {-1., -1. , -1.};

    while(-1 != (opt = getopt(argc, argv, "b:mo:s:"))) {

        switch(opt) {
            case 'b':
                GasField = optarg;
                break;
            case 'm':
                DoMatter = 1;
                break;
            case 'o':
                parse_double3(optarg, Offset);
                break;
            case 's':
                parse_double3(optarg, Size);
                break;
            default:
                usage();
        }
    }
    if(argc - optind != 3) {
            usage();
    }
    argv += optind - 1;
    char * oprefix = argv[3];
    BigFile bf = {0};
    BigBlock bb = {0};
    CIC vcic = {0};
    Ngrid = atoi(argv[2]);
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

    int d;
    for(d = 0; d < 3; d ++) {
        if(Size[d] < 0) Size[d] = BoxSize;
        ViewSize[d] = ceil(Ngrid / BoxSize * Size[d]);
    }
    for(d = 0; d < 3; d ++) {
        fprintf(stderr, "Size[%d] = %g ViewSize[%d] = %d Offset[%d] = %g\n",
             d, Size[d], d, ViewSize[d], d, Offset[d]);
    }
    if(DoMatter) {
        char * buf = alloca(strlen(oprefix) + 100);
        sprintf(buf, "%sMatter.f8", oprefix);
        domatter(&bf, buf);
    }
    if(GasField) {
        char * buf = alloca(strlen(oprefix) + 100);
        sprintf(buf, "%sGas%s.f8", oprefix, GasField);
        dogas(&bf, GasField, buf);
    }
    big_file_mpi_close(&bf, MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

