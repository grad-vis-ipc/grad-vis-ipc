# INTER-PROCESS-COMMUNICATION

ipc program will pass the output of one executable file to another executable file through pipe. The program redirects the standard input and standard output using ``dup2()``. It also uses ``fork()`` and ``exec()`` to execute the other prcesses.

## ipc.cpp
``ipc.cpp`` takes in 2 arguments: 
1. an executable file that will send data through pipe, e.g. ``./w1.out``
1. an executable file that will receive data from pipe, e.g. ``./rscode``
### run
```
make ipc1
```
```
make test1
```
### Expected output
```
=======================
[./w1.out] will send data
This is rust code. I received:
program 1 data<RUST>
====== END of execution =======
```

**P.S. I added ``<RUST>`` in the back of line of messgae to check if it's printing from ``./rscode``**

## ipc2.cpp
``ipc.cpp`` takes in 3 arguments: 
1. an executable file that will send data through pipe, e.g. ``./write.out``
1. an agument that is needed for sending program, e.g. ``4``
1. an executable file that will receive data from pipe, e.g. ``./rscode``
### run
```
make ipc2
```
```
make test2
```
### Expected output
```
=======================
[./write.out] will send data
This is rust code. I received:
hello world<RUST>
hello world<RUST>
hello world<RUST>
hello world<RUST>
====== END of execution =======
```

## ipc3.cpp
This program forks 4 child processes. Each will run ``./w1.out``, ``./w2.out``, ``./w3.out``, and ``./rscode`` respectively. ``./w1.out``, ``./w2.out`` and ``./w3.out`` will all write to the pipe, and ``./rscde`` will print all the message from pipe.

This program implements process queue using shared memory and semaphore.
### run
```
make ipc3
```
```
make test3
```
### Expected output
```
[Successfully attached shared variable to shared memory]
[./w1.out] will send data
[./w2.out] will send data
[./w3.out] will send data
[./rscode] will receive data
This is rust code. I received:
program 1 data<RUST>
program 2 data<RUST>
program 3 data<RUST>
[Successfully detach shared memory segment].
[Successfully removed shared memory segment].

```


## Clean
```
make clean
```