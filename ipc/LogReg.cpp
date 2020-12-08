// #include <time.h> /* time_t, struct tm, difftime, time, mktime */

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include "icecream.hpp"
#include "xframe/xio.hpp"
#include "xframe/xvariable.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xrandom.hpp"

#include <unistd.h>
#include <sys/shm.h> 
#include <sys/wait.h> 
#include <fcntl.h>
#include <string>
#include <vector>

using namespace std;
#define GRAD_READ pipeML3D[0]
#define GRAD_WRITE pipeML3D[1]

struct pcb{
    pid_t pid;
    pcb* next;
};


using d_vec = xt::xarray<double>;
using t = std::size_t;
using d = double;
using b_vec = xt::xarray<bool>;

struct LogReg {
  d_vec weights, features, target, ll_old;
  t max_iter;
  d learning_rate;
  LogReg(const d_vec& features, const d_vec& target, const t max_iter,
         const d learning_rate)
      : weights(xt::zeros<d>({features.shape()[1]})),
        features(features),
        max_iter(max_iter),
        target(target),
        ll_old(1e99),
        learning_rate(learning_rate) {}
  d_vec train() {
    // for early stopping
    for (t i = 0; i < max_iter; ++i) {
      auto scores = train_step();
      if (i % 1000 == 0) {
        // log likelihood should change between iterations
        auto ll = xt::sum(target * (scores)-xt::log(1 + xt::exp(scores)));
        IC(ll);
        if (xt::allclose(ll, ll_old)) {
          auto n_iterations_converged = i;
          IC(n_iterations_converged);
          break;
        } else if (xt::any(xt::greater(ll, ll_old))) {
          auto n_iterations_diverged = i;
          IC(n_iterations_diverged);
          break;
        }
        ll_old = ll;
      }
    }
    return weights;
  }
  d_vec train_step() {
    auto scores = xt::linalg::dot(features, weights);       // initial dot product
    auto predictions = 1.0 / (1.0 + xt::exp(-1 * scores));  // pipe through sigmoid
    auto error = target - predictions;                      // vector of errors made
    auto gradient = xt::linalg::dot(xt::transpose(features), error);  // gradient
    weights = weights + gradient * learning_rate;                     // update step
    return scores;
  }
  d_vec predict(const d_vec& test_set) {
    auto scores = xt::linalg::dot(xt::eval(test_set), weights);
    auto logits = 1.0 / (1.0 + xt::exp(-1 * scores));
    b_vec predictions = xt::eval(logits > .5);
    return predictions;
  }
  auto to_vec() { return std::vector<double>(weights.begin(), weights.end()); }
};

int main(const int argc, const char* argv[]) {
   if(argc != 2){
        cout << "Usage: " << argv[0] << " <readfile>" << endl;
        return EXIT_FAILURE;       
    }
    int* pids;
    int pipeML3D[2];
    int volatile *ready;
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
    ready = (int volatile *) shmat (shmid1, NULL, 0);   
    if(*ready == -1){
        cout << "Attachment errr" << endl;
        exit(1);
    }
    *ready = 1;
    if(pipe(pipeML3D) == -1 ){  
        perror("Pipe creation failure.");
        exit(EXIT_FAILURE);
    }
    
   

 
  // icecream::ic.prefix("DEBUG| ").line_wrap_width(80);
  // if (argc < 5) {
  //   std::cout << "Usage: " << argv[0] << " <train> <test> <iterations> <step>"
  //             << std::endl;
  //   return EXIT_FAILURE;
  // }
  const auto [ N_ITER, LEARNING_RATE] =
      std::make_tuple(1000, 1e-21);

  // create two classes A and B which are normally distributed
  auto A = xt::random::randn<double>({100, 3}, 1, .3);
  auto B = xt::random::randn<double>({100, 3}, -1, .3);

  // concatenate
  auto features = xt::vstack(xt::xtuple(A, B));

  // std::cout << features << std::endl;
  auto labels = xt::squeeze(
      xt::hstack(xt::xtuple(xt::ones<d>({1, 100}), xt::zeros<d>({1, 100}))));
  // std::cout << labels << std::endl;
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
      //  string data;
      
        while(true){
       

            auto predictions = clf.predict(test_set);
            // IC(predictions);

            // take a single step in the classifier
            clf.train_step();

            // print out all new predictions in form x y z LABEL
            for (int j = 0; j < 40; ++j) {
              while(*ready!=1){}
              auto row = xt::view(test_set, j, xt::all());
              auto pred = xt::view(predictions, xt::all(), j);
              auto t_vec = std::vector<double>(row.begin(), row.end());
              auto p_vec = std::vector<double>(pred.begin(), pred.end());
              std::cout << t_vec[0] << " " << t_vec[1] << " " << t_vec[2] << " " << p_vec[0] << std::endl;
            }
        }
        

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
        status = shmdt (const_cast<int*>(ready));
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
        dup2(GRAD_READ,STDIN_FILENO);  /* get input from pipe */
        execlp(argv[1],argv[1], NULL);   
   }
        

  

  return EXIT_SUCCESS;
}
