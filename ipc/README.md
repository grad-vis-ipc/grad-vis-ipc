# IPC

This program acts like a pipe between a write and a read processes using ``pipe()``, ``dup2``, ``fork()``, and ``exec()``. The read process can take a while to process the data while the write process keeps sending data to the buffer. To avoid exceeding the limit of pipe buffer capacity, this program also uses ``shmget()`` and ``shmat()``. It creates a shared memory using ``shmget()`` and attach both the read and the write processes to it using ``shmat()``. Inside the shared memory, there's an integer variable named ``ready``. The read process will set the value of ``ready`` to 1 when it's ready to receive a new data. The write process will check for the value of ``ready`` and only send data when it equals to 1, then sets the value of ``ready`` to 0 after sending one data. 

The program calls another program inside and acts like a write program as a whole. It takes in an argument, which is the executable filename of a read program. 

# Simple example

The program will print 1-99 and receive.cpp will take one data at a time.

**RUN**

```bash
make example
make test_example
```