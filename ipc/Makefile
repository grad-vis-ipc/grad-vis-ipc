all: ipc program2 program1 ipc2 w1 w2 w3 ipc3


program1: write.cpp
	g++ -o write.out write.cpp
program2: rscode.rs
	rustc rscode.rs
ipc: ipc.cpp
	g++ -o ipc.out ipc.cpp
ipc2: ipc2.cpp
	g++ -o ipc2.out ipc2.cpp
ipc3: ipc3.cpp
	g++ -o ipc3.out ipc3.cpp -pthread
w1: w1.cpp
	g++ -o w1.out w1.cpp
w2: w2.cpp
	g++ -o w2.out w2.cpp
w3: w3.cpp
	g++ -o w3.out w3.cpp
test1:
	./ipc.out ../grad/dispatcher.out ./rscode
test2:
	./ipc2.out ./write.out 4 ./rscode
test3: 
	./ipc3.out 
clean:
	rm -f *.out 
	rm -f rscode
