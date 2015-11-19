OBOE=$(pwd)/OBOE/dist
CXX_COMPILER=g++-4.9
C_COMPILER=gcc-4.9

rm -r build
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=DEBUG  -DOBOE=${OBOE} -DLAPACKPP=${OBOE}/lapackpp -DCMAKE_VERBOSE_MAKEFILE=ON ..

make
cd ..
