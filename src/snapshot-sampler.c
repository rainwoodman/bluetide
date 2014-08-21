#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include "bigfile-mpi.h"

void usage() {
    fprintf(stderr, "usage: bigfile-copy-mpi [-n Nfile] [-s seed] [-r ratio] filepath block newblock\n");
    exit(1);

}
#define DONE_TAG 1293
#define ERROR_TAG 1295
#define DIE_TAG 1290
#define WORK_TAG 1291

MPI_Datatype MPI_TYPE_WORK;
BigFile bf = {0};
BigBlock bb = {0};
BigBlock bbnew = {0};
int verbose = 0;
int Nfile = -1;
size_t buffersize = 256 * 1024 * 1024;
int ThisTask, NTask;
void slave(void);
void server(void);
int SEED = 1979;
double RATIO;
int EVERY = 1;

struct work {
    int64_t offset;
    int64_t chunksize;
    int64_t offsetnew;
    int64_t chunksizenew;
    struct drand48_data randstate;
};

int main(int argc, char * argv[]) {

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    MPI_Type_contiguous(sizeof(struct work), MPI_BYTE, &MPI_TYPE_WORK);
    MPI_Type_commit(&MPI_TYPE_WORK);

    int ch;
    while(-1 != (ch = getopt(argc, argv, "n:vb:s:r:"))) {
        switch(ch) {
            case 's':
                SEED = atoi(optarg);
                break;
            case 'r':
                RATIO = atof(optarg);
                break;
            case 'n':
                Nfile = atoi(optarg);
                break;
            case 'b':
                sscanf(optarg, "%td", &buffersize);
                break;
            case 'v':
                verbose = 1;
                break;
            default:
                usage();
        }
    }
    if(argc - optind + 1 != 4) {
        usage();
    }
    EVERY = 1 / RATIO;
    argv += optind - 1;
    if(0 != big_file_mpi_open(&bf, argv[1], MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        exit(1);
    }
    if(0 != big_file_mpi_open_block(&bf, &bb, argv[2], MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        exit(1);
    }
    if(Nfile == -1 || bb.Nfile == 0) {
        Nfile = bb.Nfile;
    }

    if(0 != big_file_mpi_create_block(&bf, &bbnew, argv[3], bb.dtype, bb.nmemb, Nfile, bb.size, MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to create temp: %s\n", big_file_get_error_message());
        exit(1);
    }

    if(bbnew.size != bb.size) {
        abort();
    }

    /* copy attrs */
    size_t nattr;
    BigBlockAttr * attrs = big_block_list_attrs(&bb, &nattr);
    int i;
    for(i = 0; i < nattr; i ++) {
        BigBlockAttr * attr = &attrs[i];
        big_block_set_attr(&bbnew, attr->name, attr->data, attr->dtype, attr->nmemb);
    }

    big_block_set_attr(&bbnew, "RATIO", &RATIO, "f8", 1);
    big_block_set_attr(&bbnew, "SEED", &SEED, "i8", 1);
    if(bb.nmemb > 0 && bb.size > 0) {
    /* copy data */
        if(ThisTask == 0) {
            server();
        } else {
            slave();
        }
    }
    if(0 != big_block_mpi_close(&bbnew, MPI_COMM_WORLD)) {
        fprintf(stderr, "failed to close new: %s\n", big_file_get_error_message());
        exit(1);
    }
    big_block_mpi_close(&bb, MPI_COMM_WORLD);
    big_file_mpi_close(&bf, MPI_COMM_WORLD);
    return 0;
}

void server() {
    int64_t offset = 0;
    int64_t offsetnew = 0;
    struct work work;
    struct drand48_data randstate;

    srand48_r(SEED, &randstate);

    for(offset = 0; offset < bb.size; ) {
        int64_t chunksize = buffersize / (bb.nmemb * dtype_itemsize(bb.dtype));
        MPI_Status status;
        int result = 0;
        MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                &status);
        if(status.MPI_TAG == ERROR_TAG) {
            break;
        }
        /* never read beyond my end (read_simple caps at EOF) */
        if(offset + chunksize > bb.size) {
            chunksize = bb.size - offset;
        }

        int64_t i;
        memcpy(&work.randstate, &randstate, sizeof(randstate));

        int64_t chunksizenew = 0;
        for(i = 0; i < chunksize; i ++) {
            double v;
            drand48_r(&randstate, &v);
            if(v < RATIO) {
                chunksizenew ++;
            }
        }
        
        work.offset = offset;
        work.chunksize = chunksize;
        work.offsetnew = offsetnew;
        work.chunksizenew = chunksizenew;

        MPI_Send(&work, 1, MPI_TYPE_WORK, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);

        offset += chunksize;
        offsetnew += chunksizenew;
        if(verbose) {
            fprintf(stderr, "%td / %td done (%0.4g%%)\r", offset, bb.size, (100. / bb.size) * offset);
        }
    }
    int i;
    for(i = 1; i < NTask; i ++) {
        struct work work;
        MPI_Send(&work, 1, MPI_TYPE_WORK, i, DIE_TAG, MPI_COMM_WORLD);
    }

}
void slave() {
    int result = 0;
    MPI_Send(&result, 1, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);

    while(1) {
        struct work work;
        MPI_Status status;
        MPI_Recv(&work, 1, MPI_TYPE_WORK, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if(status.MPI_TAG == DIE_TAG) {
            break;
        }
        int64_t offset = work.offset;
        int64_t chunksize = work.chunksize;
        int64_t offsetnew = work.offsetnew;
        int64_t chunksizenew = work.chunksizenew;

        BigArray array;
        BigBlockPtr ptrnew;
        BigArray arraynew;

        size_t dims[2] = {chunksize, bb.nmemb};
        char * datanew = malloc(dtype_itemsize(array.dtype) * array.dims[1] * chunksizenew);
        big_array_init(&arraynew, datanew, array.dtype, 2, dims, NULL); 

        if(0 != big_block_read_simple(&bb, offset, chunksize, &array, NULL)) {
            fprintf(stderr, "failed to read original: %s\n", big_file_get_error_message());
            result = -1;
            goto bad;
        }
        if(array.dims[0] != chunksize) {
            abort();
        }
        {
            char * p = array.data;
            char * q = datanew;

            size_t elsize = dtype_itemsize(array.dtype) * array.dims[1];
            size_t count = 0;
            ptrdiff_t i;
            for(i = 0; i < array.dims[0]; i ++) {
                double v;
                drand48_r(&work.randstate, &v);
                if(v < RATIO) {
                    memcpy(q, p, elsize);
                    q += elsize;
                    count ++;
                }
                p += elsize;
            }
            if(count != chunksizenew) abort();
        }
        free(array.data);
        if(0 != big_block_seek(&bbnew, &ptrnew, offsetnew)) {
            fprintf(stderr, "failed to seek new: %s\n", big_file_get_error_message());
            result = -1;
            free(datanew);
            goto bad;
        }

        if(0 != big_block_write(&bbnew, &ptrnew, &array)) {
            fprintf(stderr, "failed to write new: %s\n", big_file_get_error_message());
            result = -1;
            free(datanew);
            goto bad;
        }

        free(array.data);
        MPI_Send(&result, 1, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);
        continue;
    bad:
        MPI_Send(&result, 1, MPI_INT, 0, ERROR_TAG, MPI_COMM_WORLD);
        continue;
    }
    return;
}
