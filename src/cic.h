
typedef struct {
    /* aligned at 8 bytes on 64bit machines; this is ugly shall replace with
     * int64 etc because it is used in IO. */
    size_t Ngrid[3];
    double BoxSize[3];
    int periodic;

    /* private */
    size_t size;
    double CellSize[3];
    size_t strides[3];
    double * buffer;
} CIC;

void cic_init(CIC * cic, int Ngrid[3], double BoxSize[3], int periodic);
void cic_add_particle(CIC * cic, double Pos[3], double mass);
void cic_destroy(CIC * cic);
