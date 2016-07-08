ifeq ($(DEBUG), 1)
	flags := -g -G
else
	flags := -O4
endif

sstsoft: sstsoft.o 
	nvcc -ccbin g++ $(flags) -o sstsoft -lcublas sstsoft.o

sstsoft.o: sstsoft.cu
	nvcc -ccbin g++ -dc $(flags) -o sstsoft.o sstsoft.cu

clean:
	rm sstsoft sstsoft.o
