default: write.cpp receive.cpp test.cpp
	g++ -o write.out write.cpp
	g++ -o receive.out receive.cpp
	g++ -o test.out test.cpp -pthread
test: 
	./test.out ./write.out ./receive.out

clean:
	rm -f *.out 

