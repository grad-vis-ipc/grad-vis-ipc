FLAGS = -std=c++17 -lblas -fsanitize=address -O -march=native


log_reg.out: logistic_regression.o reg_driver.cpp
	g++ logistic_regression.o reg_driver.cpp $(FLAGS) -o log_reg.out

logistic_regression.o: logistic_regression.cpp
	g++ -c logistic_regression.cpp $(FLAGS)

test1: log_reg.out
	./log_reg.out monks-1.train monks-1.test 4000 0.0005

test2: log_reg.out
	./log_reg.out monks-2.train monks-2.test 8300 0.001

test3: log_reg.out
	./log_reg.out monks-3.train monks-3.test 8300 0.001

sklearn1:
	./log_reg.py monks-1.train monks-1.test 

sklearn2:
	./log_reg.py monks-2.train monks-2.test 

sklearn3:
	./log_reg.py monks-3.train monks-3.test 

clean:
	rm -f *.o