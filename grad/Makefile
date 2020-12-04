FLAGS = -std=c++17 -lblas -fsanitize=address #-O -march=native
MONK_FILE_BASE = data_monks/monks-
L_RATE?=.001
MX_ITER?=1000000
D_SET?=1

SRC?=dispatcher.cpp
file: 
	g++ $(SRC) $(FLAGS) -o $(SRC).out
	./$(SRC).out

log_reg.out: logistic_regression.o reg_driver.cpp
	g++ logistic_regression.o reg_driver.cpp $(FLAGS) -o log_reg.out

svm.out: support_vector_machine.o support_vector_driver.cpp
	g++ support_vector_machine.o support_vector_driver.cpp $(FLAGS) -o svm.out

dispatcher.out: dispatcher.cpp
	g++ dispatcher.cpp $(FLAGS) -o dispatcher.out

kern_log_reg.out: logistic_regression.o kernel_reg_driver.cpp
	g++ logistic_regression.o kernel_reg_driver.cpp $(FLAGS) -o kern_log_reg.out

logistic_regression.o: logistic_regression.cpp
	g++ -c logistic_regression.cpp $(FLAGS)

support_vector_machine.o: support_vector_machine.cpp
	g++ -c support_vector_machine.cpp $(FLAGS)

svm: svm.out
	./svm.out $(MONK_FILE_BASE)1.train $(MONK_FILE_BASE)1.test $(MX_ITER) $(L_RATE)

pearson: kern_log_reg.out
	./kern_log_reg.out $(MONK_FILE_BASE)$(D_SET).train $(MONK_FILE_BASE)$(D_SET).test $(MX_ITER) $(L_RATE)

kern: kern_log_reg.out
	./kern_log_reg.out $(MONK_FILE_BASE)$(D_SET).train $(MONK_FILE_BASE)$(D_SET).test $(MX_ITER) $(L_RATE)

test: log_reg.out
	./log_reg.out $(MONK_FILE_BASE)$(D_SET).train $(MONK_FILE_BASE)$(D_SET).test $(MX_ITER) $(L_RATE)

sklearn:
	./log_reg.py $(MONK_FILE_BASE)$(D_SET).train $(MONK_FILE_BASE)$(D_SET).test 

clean:
	rm -f *.o *.out