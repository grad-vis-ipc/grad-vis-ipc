#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <unistd.h>
using namespace std;

int main(){

	int count = 1;
	while(count <100){

		cout << count<< endl;

		count++;
	}

}