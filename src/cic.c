#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "cic.h"

void cic_init(CIC * cic, double BoxSize, int Ngrid, int ViewSize[3]) {
    int k;
    size_t last = 1;
    cic->CellSize = BoxSize / Ngrid;
    cic->BoxSize = BoxSize;
    for(k = 0; k < 3; k++) {
        cic->strides[2 - k] = last;
        last *= ViewSize[2 - k];
        cic->ViewSize[2 - k] = ViewSize[2 - k];
    }
    cic->Ngrid = Ngrid;
    cic->buffer = calloc(last, sizeof(double));
    cic->size = last;
}

void cic_add_particle(CIC * cic, double Pos[3], double mass) {
    ptrdiff_t ret = 0;
    int k;
    int iCell[3];
    double Res[3];
    /* do periodic box here */
    for(k = 0; k < 3; k++) {
        double tmp = Pos[k] / cic->CellSize;
        iCell[k] = floor(tmp);
        Res[k] = tmp - iCell[k];
        while(iCell[k] < 0) iCell[k] += cic->Ngrid;
        while(iCell[k] >= cic->Ngrid) iCell[k] -= cic->Ngrid;
    }

    int connection = 0;
    double wtsum = 0.0;
    for(connection = 0; connection < 8; connection++) {
        double weight = 1.0;
        ptrdiff_t linear = 0;
        int out = 0;
        for(k = 0; k < 3; k++) {
            int offset = (connection >> k) & 1;
            int tmp = iCell[k] + offset;
            if(tmp >= cic->Ngrid) {
                tmp -= cic->Ngrid;
            }
            if(tmp < 0) {
                tmp += cic->Ngrid;
            }
            if(tmp >= cic->ViewSize[k]) {
                out = 1;
            }
            
            linear += tmp * cic->strides[k];
            weight *= offset?
                /* offset == 1*/ (Res[k])    :
                /* offset == 0*/ (1 - Res[k]);
        }
        wtsum += weight;
        if(out) continue;
#pragma omp atomic
        cic->buffer[linear] += weight * mass;
    }
    if(fabs(wtsum - 1.0) > 1e-6) {
        abort(); 
    }
}
void cic_destroy(CIC * cic) {
    free(cic->buffer);
    cic->buffer = NULL;
}



