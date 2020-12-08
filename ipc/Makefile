FLAGS = -I "xtensor-blas/include" -I "xtl/include" -I "xframe/include" -I "xtensor/include" -I "xframe/include" -std=c++17 -fsanitize=address #-O -march=native


default: sendgrad_receive.cpp receive.cpp 
	g++ -o sendgrad_receive.out  sendgrad_receive.cpp 
	g++ -o receive.out receive.cpp
example: sendcount_receive.cpp receive.cpp
	g++ -o receive.out receive.cpp
	g++ -o sendcount_receive.out sendcount_receive.cpp

log_reg: LogReg.cpp
	g++ -o LogReg.out ${FLAGS} LogReg.cpp -lblas 
	g++ -o receive.out receive.cpp

test_log: LogReg.out receive.out
	./LogReg.out ./receive.out
demo: sendgrad_receive.out
	./sendgrad_receive.out ./receive.out

test: 
	./sendgrad_receive.out ./receive.out
	
test_example: 
	./sendcount_receive.out ./receive.out

clean:
	rm -f *.out 



