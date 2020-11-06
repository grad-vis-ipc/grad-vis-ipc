/*
    g++ -o write write.cpp
    g++ -o read read.cpp
    g++ -o ipc2 ipc2.cpp
    ./ipc2 ./write 3 ./read

    rustc rscode.rs
    g++ -o read read.cpp
    g++ -o ipc2 ipc2.cpp
    ./ipc2 ./write 3 ./rscode
*/
#include <stdio.h> // standard input and output
#include <stdlib.h>  /* POSIX operating system API */
#include <unistd.h>  
#include <sys/types.h> /*pthread*/
#include <sys/wait.h>  /* wait */
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <string>
using namespace std;

int pipeML3D[2];

void Send (char * program1, char*argv[]){
    close(pipeML3D[0]);
    dup2(pipeML3D[1], STDOUT_FILENO); /* make output go to pipe */
    execv(program1, argv+1);

}
int main(int argc, char* argv[]){
	int status;     
	if(pipe(pipeML3D) == -1 ){  
        cout <<"Pipe creation failure " << endl;
        exit(1);
    }
    int pid = fork();
    if(pid == -1){
    	// error
    	return -1;
    }
    else if(pid == 0){ 
        Send(argv[1],  argv);
    	//https://www.youtube.com/watch?v=pO1wuN3hJZ4
    }
    pid = fork();
    if(pid == -1){
    	return -1;
    }
    else if(pid == 0){ 
    	close(pipeML3D[1]);
    	dup2(pipeML3D[0], STDIN_FILENO); /* get input from pipe */
    	execlp(argv[3], argv[3],NULL);
    
    }
    close(pipeML3D[0]);
    close(pipeML3D[1]);
    waitpid(-1, &status, 0);
    waitpid(-1, &status, 0);
}
   


    

// pipe between 2 program
  //https://stackoverflow.com/questions/8723628/get-the-output-of-one-c-program-into-a-variable-in-another-c-program

//continuously
//https://stackoverflow.com/questions/40936508/using-pipe-to-pass-data-between-two-processes-continuously


//https://www.unix.com/programming/58138-c-how-use-pipe-fork-stdin-stdout-another-program.html


//pipe
//http://web.stanford.edu/~hhli/CS110Notes/CS110NotesCollection/Topic%202%20Multiprocessing%20(3).html

/*
pipe - Returns a pair of file descriptors, so that what's written to one can be read from the other.

fork - Forks the process to two, both keep running the same code.

dup2 - Renumbers file descriptors. With this, you can take one end of a pipe and turn it into stdin or stdout.

exec - Stop running the current program, start running another, in the same process.
*/