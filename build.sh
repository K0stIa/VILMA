OBOE=$(pwd)/OBOE/dist
# CXX_COMPILER=g++-4.7

rm -r build
mkdir build
cd build


#if  [['$local_host' -eq 'Darwin']]; then
#echo "Running on mac os"
cmake -DCMAKE_BUILD_TYPE=DEBUG  -DOBOE=${OBOE} -DLAPACKPP=${OBOE}/lapackpp -DCMAKE_VERBOSE_MAKEFILE=ON ..
#else
#echo "Running on cmpgrid"
#cmake -DOBOE=/home.dokt/antonkos/lib/oboe -DLAPACKPP=/home.dokt/antonkos/lib/lapackpp -DCMAKE_CXX_COMPILER=${CXX_COMPILER} ..
#fi	
make
cd ..
