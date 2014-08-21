all: \
    snapshot-cic \
    snapshot-sampler \
    sampler \
	$(NULL)

INCBIGFILE=../bigfile/src

snapshot-cic: snapshot-cic.c cic.c ../bigfile/src/bigfile-mpi.c ../bigfile/src/bigfile.c
	$(CC) -o $@ $^ $(LIBBIGFILE) -I$(INCBIGFILE)
sampler: sampler.c
	$(CC) -o $@ $^ 
snapshot-sampler: snapshot-sampler.c ../bigfile/src/bigfile-mpi.c ../bigfile/src/bigfile.c
	$(CC) -o $@ $^ $(LIBBIGFILE) -I$(INCBIGFILE)
