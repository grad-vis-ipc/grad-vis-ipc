FLAGS = -I "xtensor-blas/include" -I "xtl/include" -I "xframe/include" -I "xtensor/include" -I "xframe/include"-std=c++17 -lblas -fsanitize=address -lcblas #-O -march=native


default: sendgrad_receive.cpp receive.cpp 
	g++ -o sendgrad_receive.out ${FLAGS} sendgrad_receive.cpp 
	g++ -o receive.out receive.cpp
example: sendcount_receive.cpp receive.cpp
	g++ -o receive.out receive.cpp
	g++ -o sendcount_receive.out sendcount_receive.cpp

demo: sendgrad_receive.out
	./sendgrad_receive.out ./receive.out

test: 
	./sendgrad_receive.out ./receive.out
	
test_example: 
	./sendcount_receive.out ./receive.out

clean:
	rm -f *.out 

