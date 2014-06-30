all: \
    snapshot-cic
	$(NULL)
LIBBIGFILE=$(HOME)/bigfile/src/bigfile.a \
        $(HOME)/bigfile/src/bigfile-mpi.a
INCBIGFILE=$(HOME)/bigfile/src

snapshot-cic: snapshot-cic.c  cic.c
	$(CC) -o $@ $^ $(LIBBIGFILE) -I$(INCBIGFILE)
