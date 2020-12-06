default: sendcount_receive.cpp receive.cpp
	g++ -o receive.out receive.cpp
	g++ -o sendcount_receive.out sendcount_receive.cpp
sr1: sendgrad_receive.cpp receive.cpp 
	g++ -o sendgrad_receive.out -I "xtensor/include" -I "xtl/include" -std=c++17 -lblas -fsanitize=address sendgrad_receive.cpp   
	g++ -o receive.out receive.cpp
test1: 
	./sendgrad_receive.out ./receive.out
test: 
	./sendcount_receive.out ./receive.out

clean:
	rm -f *.out 

