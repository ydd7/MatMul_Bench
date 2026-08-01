how to compile:

clang++ -O3 -std=c++17 bench.cpp -o macbench -framework Accelerate

or

clang++ -O3 -Ofast -march=native -std=c++17 bench.cpp -o macbench -framework Accelerate
