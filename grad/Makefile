FLAGS = -std=c++17 -lblas -fsanitize=address -O -march=native
MONK_FILE_BASE = data_monks/monks-
L_RATE?=.001

log_reg.out: logistic_regression.o reg_driver.cpp
	g++ logistic_regression.o reg_driver.cpp $(FLAGS) -o log_reg.out

kern_log_reg.out: logistic_regression.o kernel_reg_driver.cpp
	g++ logistic_regression.o kernel_reg_driver.cpp $(FLAGS) -o kern_log_reg.out

logistic_regression.o: logistic_regression.cpp
	g++ -c logistic_regression.cpp $(FLAGS)

kern1: kern_log_reg.out
	./kern_log_reg.out data_monks/monks-1.train data_monks/monks-1.test 4800000 0.002

pearson1: kern_log_reg.out
	./kern_log_reg.out data_monks/monks-1.train data_monks/monks-1.test 4800000 $(L_RATE)

pearson2: kern_log_reg.out
	./kern_log_reg.out data_monks/monks-2.train data_monks/monks-2.test 4800000 $(L_RATE)

pearson3: kern_log_reg.out
	./kern_log_reg.out data_monks/monks-3.train data_monks/monks-3.test 4800000 $(L_RATE)

kern2: kern_log_reg.out
	./kern_log_reg.out data_monks/monks-2.train data_monks/monks-2.test 4800000 0.0002

kern3: kern_log_reg.out
	./kern_log_reg.out data_monks/monks-3.train data_monks/monks-3.test 4800000 0.0003

test1: log_reg.out
	./log_reg.out $(MONK_FILE_BASE)1.train $(MONK_FILE_BASE)1.test 4000 0.0005

test2: log_reg.out
	./log_reg.out $(MONK_FILE_BASE)2.train $(MONK_FILE_BASE)2.test 8300 0.001

test3: log_reg.out
	./log_reg.out $(MONK_FILE_BASE)3.train $(MONK_FILE_BASE)3.test 8300 0.001

sklearn1:
	./log_reg.py $(MONK_FILE_BASE)1.train $(MONK_FILE_BASE)1.test 

sklearn2:
	./log_reg.py $(MONK_FILE_BASE)2.train $(MONK_FILE_BASE)2.test 

sklearn3:
	./log_reg.py $(MONK_FILE_BASE)3.train $(MONK_FILE_BASE)3.test 

clean:
	rm -f *.o *.out