project(sparse_sgd)
cmake_minimum_required(VERSION 2.8)

#if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
#SET(CMAKE_CXX_COMPILER "g++-4.7.3")
#SET(CMAKE_CC_COMPILER "gcc-4.7.3")
#endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

include_directories(.)
include_directories(${OBOE}/include)
include_directories(${LAPACKPP}/include)

LINK_DIRECTORIES(${LAPACKPP}/lib)

aux_source_directory(bmrm BMRM)
aux_source_directory(oracle ORACLE)
aux_source_directory(solver SOLVER)
aux_source_directory(. THIS)

set( SRC data.h		dense_vector.hpp  model_evaluator.h    sparse_matrix.h	  sparse_vector.h    vector_interface.h
dense_vector.h	loss.h		  model_evaluator.hpp  sparse_matrix.hpp  sparse_vector.hpp data.cpp)

add_executable(main main.cpp ${BMRM} ${ORACLE} ${SOLVER} ${SRC})
target_link_libraries(main lapack)
target_link_libraries(main blas)
target_link_libraries(main lapackpp)
target_link_libraries(main ${OBOE}/lib/liboboe.a)

add_executable(report_resuts report_resuts.cpp ${BMRM} ${ORACLE} ${SRC})
target_link_libraries(report_resuts lapack)
target_link_libraries(report_resuts blas)
target_link_libraries(report_resuts lapackpp)
target_link_libraries(report_resuts ${OBOE}/lib/liboboe.a)

add_executable(train_vilma train_vilma.cpp ${BMRM} ${ORACLE} ${SRC})
target_link_libraries(train_vilma lapack)
target_link_libraries(train_vilma blas)
target_link_libraries(train_vilma lapackpp)
target_link_libraries(train_vilma ${OBOE}/lib/liboboe.a)

add_executable(test_vilma test_vilma.cpp ${BMRM} ${ORACLE} ${SRC})
target_link_libraries(test_vilma lapack)
target_link_libraries(test_vilma blas)
target_link_libraries(test_vilma lapackpp)
target_link_libraries(test_vilma ${OBOE}/lib/liboboe.a)
