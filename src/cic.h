
typedef struct {
    int ViewSize[3];
    int Ngrid;
    double BoxSize;

    /* private */
    size_t size;
    double CellSize;
    size_t strides[3];
    double * buffer;
} CIC;

void cic_init(CIC * cic, double BoxSize, int Ngrid, int ViewSize[3]);
void cic_add_particle(CIC * cic, double Pos[3], double mass);
void cic_destroy(CIC * cic);
