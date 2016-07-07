sstsoft: sstsoft.o 
	nvcc -ccbin g++ -g -G -o sstsoft -lcublas /home/speverel/speverel_home/cudpp/lib/libcudpp64d.a sstsoft.o

sstsoft.o: sstsoft.cu
	nvcc -ccbin g++ -dc -g -G -o sstsoft.o sstsoft.cu

clean:
	rm sstsoft sstsoft.o
