#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include <signal.h>

void usage() {
    fprintf(stderr, "usage: sampler [-s SEED] [-a ITEM ALIGNMENT bytes (4)] ratio\n");
    exit(1);
}

size_t Align = 4;
double Ratio = 1.0;
int THRESH;
const int BUFFERSIZE = 32 * 1024 * 1024;
size_t countin = 0;
size_t countout = 0;
void usr1(int i) {
    fprintf(stderr, "sampler[%d] %td items in %td items out item size %td (r=%g ~ %g)\n",
        getpid(), countin, countout, Align, countout * 1.0 / countin, Ratio);
}
int main(int argc, char * argv[]) {
    char opt;
    while(-1 != (opt = getopt(argc, argv, "s:a:"))) {
        switch(opt) {
            case 's':
                srand(atoi(optarg));
                break;
            case 'a':
                Align = atoi(optarg);
                break;
            default:
                usage();
        }
    }
    if(argc - optind != 1) {
        usage();
    }
    argv += optind - 1;
    Ratio = atof(argv[1]);
    THRESH = RAND_MAX * Ratio;
    signal(SIGUSR1, usr1);
    char * buffer = malloc(BUFFERSIZE);
    char * outbuffer = malloc(BUFFERSIZE);
    size_t counts = BUFFERSIZE / Align;
    
    while(!feof(stdin)) {
        ptrdiff_t readcount = fread(buffer, Align, counts, stdin);
        ptrdiff_t i;
        char * p = outbuffer;
        char * q = buffer;
        for(i = 0; i < readcount; i ++) {
            int r = rand();
            countin ++;
            if(r <= THRESH) {
                memcpy(p, q, Align);
                p += Align;
                countout ++;
            }
            q += Align;
        }
        fwrite(outbuffer, 1, p - outbuffer, stdout);
        fflush(stdout);
    }
    free(buffer);
    free(outbuffer);
    return 0;
}
