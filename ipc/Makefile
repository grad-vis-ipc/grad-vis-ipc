all: program1 program2 ipc

program1: write.cpp
	g++ -o write.out write.cpp
program2: rscode.rs
	rustc rscode.rs
ipc: ipc.cpp
	g++ -o ipc.out ipc.cpp

test:
	./ipc.out ./write.out 4 ./rscode
clean:
	rm -f *.out 
	rm -f rscode