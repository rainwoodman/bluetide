all: \
    snapshot-cic
	$(NULL)

INCBIGFILE=../bigfile/src

snapshot-cic: snapshot-cic.c cic.c ../bigfile/src/bigfile-mpi.c ../bigfile/src/bigfile.c
	$(CC) -o $@ $^ $(LIBBIGFILE) -I$(INCBIGFILE)
