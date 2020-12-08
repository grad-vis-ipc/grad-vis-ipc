#include <stdio.h> 
#include <sys/wait.h>  
#include <iostream>
#include <unistd.h>
#include <mutex>
#include <sys/shm.h> 
#include <semaphore.h>
#include <fcntl.h>
#include <string>
#include <vector>
#include <signal.h>
#include <time.h> /* time_t, struct tm, difftime, time, mktime */
#include <chrono>
#include <fstream>
#include <thread>
#include "icecream.hpp"
#include "xtensor/xadapt.hpp"
#include "../grad/LogReg.cpp"
using namespace std;

#define GRAD_READ pipeML3D[0]
#define GRAD_WRITE pipeML3D[1]



struct pcb{
    pid_t pid;
    pcb* next;
};

int main(int argc, char* argv[]){
    if(argc != 2){
        cout << "Usage: " << argv[0] << " <readfile>" << endl;
        return EXIT_FAILURE;       
    }
    int* pids;
    int pipeML3D[2];
    int *ready;
    int shmid1, shmid2;  
    int i;
    pid_t receive_pid;
    int status;  
    pcb * head;   
    pid_t pid;
    key_t key = ftok("shmfile",65); 
    shmid1 = shmget (key, sizeof (int), IPC_CREAT | 0666);
    if (shmid1 < 0) {                           /* shmget error check */
        cout << "shmget error" << endl;
        exit (1);
    } 
    /* Attach shared variable to shared memory using shmat  */
    ready = (int *) shmat (shmid1, NULL, 0);   
    if(*ready == -1){
        cout << "Attachment errr" << endl;
        exit(1);
    }
    *ready = 0;
    if(pipe(pipeML3D) == -1 ){  
        perror("Pipe creation failure.");
        exit(EXIT_FAILURE);
    }
    
   
    pid = fork();
    if (pid < 0) {       /* Error check */    
        cout << "fork error" << endl;
        exit(1);
    }
    else{
       receive_pid = pid;
    }
    
    if (pid > 0){ /* Parent Process */ 
  
        dup2(GRAD_WRITE , STDOUT_FILENO); /* make output go to pipe */
    	close(GRAD_READ);


    	/*********************************************************************/
		/***************** Machine Learning Code *****************************/
		/*********************************************************************/
        const auto [N_ITER, LEARNING_RATE] =
        std::make_tuple( 1000, 1e-19);

        // create two classes A and B which are normally distributed
        auto A = xt::random::randn<double>({100, 3}, 1, .3);
        auto B = xt::random::randn<double>({100, 3}, -1, .3);

        // concatenate
        auto features = xt::vstack(xt::xtuple(A, B));

        auto labels = xt::squeeze(
          xt::hstack(xt::xtuple(xt::ones<d>({1, 100}), xt::zeros<d>({1, 100}))));
    
        // IC(labels);
        auto e_features = xt::eval(features);
        xt::random::seed(0);
        xt::random::shuffle(e_features);

        auto e_labels = xt::eval(labels);
        xt::random::seed(0);
        xt::random::shuffle(e_labels);

        auto A_test = xt::random::randn<double>({20, 3}, 1, .5);
        auto B_test = xt::random::randn<double>({20, 3}, -1, .5);
        auto test_set = xt::vstack(xt::xtuple(A_test, B_test));

        auto clf = LogReg(e_features, e_labels, N_ITER, LEARNING_RATE);
        // auto weights  = clf.train();
        // for (int i = 0; i < 10; ++i) {
        // auto predictions = clf.predict(test_set);
        // IC(predictions);
        // clf.train_step();
        // for (int j = 0; j < 200; ++j) {
        //   auto row = xt::view(test_set, j, xt::all());
        //   auto pred = xt::view(predictions, xt::all(), j);
        //   auto t_vec = std::vector<double>(row.begin(), row.end());
        //   auto p_vec = std::vector<double>(pred.begin(), pred.end());
        //   std::cout << t_vec[0] << " " << t_vec[1] << " " << t_vec[2] << " " << p_vec[0] << std::endl;
        // }
        // }
      
        while (true) { 
            auto predictions = clf.predict(test_set);
            IC(predictions);
            clf.train_step();
            for (int j = 0; j < 40; ++j) {
                while(*ready!=1){};
                auto row = xt::view(test_set, j, xt::all());
                auto pred = xt::view(predictions, xt::all(), j);
                auto t_vec = std::vector<double>(row.begin(), row.end());
                auto p_vec = std::vector<double>(pred.begin(), pred.end());
                std::cout << t_vec[0] << " " << t_vec[1] << " " << t_vec[2] << " " << p_vec[0] << std::endl;
                *ready = 0;
            }
           
        }
        /*********************************************************************/
		/***********End of Machine Learning Code *****************************/
		/*********************************************************************/

        close(GRAD_WRITE);
        close(GRAD_READ);
 
        /* wait until read finishes reading the rest of the data */
        while((status = waitpid(receive_pid, &status, WNOHANG|WUNTRACED))) {
        	if(status == receive_pid )
        		break;
            else if(status==-1)
                return -1;
            else if(status == 0){
            	sleep(1);
            }
        }
      	
        /* deatach shared memory */
        status = shmdt (ready);
        if(status == -1){
            cout << "Error: Remove shared memory segment  " << endl;
            exit(1);
        }   
        status = shmctl (shmid1, IPC_RMID, 0); 
        if(status == -1){
            cout << "Error: Remove shared memory segment  " << endl;
            exit(1);
        }   
        exit(0);
    }
    else{   /* Child process */  
  		close(GRAD_WRITE);
        dup2(GRAD_READ,STDIN_FILENO); /* get input from pipe */
        execlp(argv[1],argv[1], NULL);   
   }
        
}
   

