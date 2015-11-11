OBOE=$(pwd)/OBOE/dist
# CXX_COMPILER=g++-4.7

rm -r build
mkdir build
cd build


#if  [['$local_host' -eq 'Darwin']]; then
#echo "Running on mac os"
cmake -DOBOE=${OBOE} -DLAPACKPP=${OBOE}/lapackpp ..
#else
#echo "Running on cmpgrid"
#cmake -DOBOE=/home.dokt/antonkos/lib/oboe -DLAPACKPP=/home.dokt/antonkos/lib/lapackpp -DCMAKE_CXX_COMPILER=${CXX_COMPILER} ..
#fi	
make
cd ..
