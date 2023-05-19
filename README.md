[![Build status](https://github.com/ib-proj/sgd_matrix_factorization/actions/workflows/build.yaml/badge.svg)](https://github.com/ib-proj/sgd_matrix_factorization/actions)

# Project

This project contains our work for implemeting the paper A Fast Parallel Stochastic Gradient Method for Matrix Factorization in Shared Memory Systems, 2016 by W.-S. Chin, Y. Zhuang, Y.-C. J. and Lin., C.-J.

# Prerequisites
* Python
* CMake 3.25 or higher.
* A C++ compiler that supports C++17 (e.g., GCC 7 or higher).
* A system with multithreading support (i.e., POSIX threads).

# Getting Started

First, clone this repository to your local machine using git:
```
git clone https://github.com/yourusername/project.git
```

After cloning the repository, navigate to the project's root directory:
```
cd project
```


# Building the Project

To build the project, follow the steps below:

* Generate the Makefile using cmake:
```
cmake .
```

* Build the project using make:
go
Copy code
```
make
```

# Running the Project

After building the project, an executable named project will be generated in your directory. You can run the project using:
```
./project
```

This will execute the program and you should see the output in your terminal.

Contributors:

- Imad Boudroua
- Nick Jofrein Tedonze

# Plot resuts


* plot results
```
python plot.py
```

![alt text](https://github.com/ib-proj/sgd_matrix_factorization/blob/main/results/rmse_over_time.png?raw=true)


# Clean code

* clean executable
```
make clean
```

* clean generated Makefile and cache
```
make clean-all
```