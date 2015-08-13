rm -r build
mkdir build
cd build
local_host=`uname -s`
echo "local host: $local_host"

CXX_COMPILER=g++-4.7

#if  [['$local_host' -eq 'Darwin']]; then
#echo "Running on mac os"
# cmake -DOBOE=/Users/Shared/research/code/python/jmlr/oboe -DLAPACKPP=/Users/Shared/research/code/python/jmlr/lapackpp ..
#else
#echo "Running on cmpgrid"
cmake -DOBOE=/home.dokt/antonkos/lib/oboe -DLAPACKPP=/home.dokt/antonkos/lib/lapackpp -DCMAKE_CXX_COMPILER=${CXX_COMPILER} ..
#fi	
make
cd ..
