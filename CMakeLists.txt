project(VILMA)
cmake_minimum_required(VERSION 2.8)

#if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
# SET(CMAKE_CXX_COMPILER "g++-4.9")
# SET(CMAKE_CC_COMPILER "gcc-4.9")
#endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(mex)

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

set(OBOE_LIBS ${OBOE}/lib/libaccpm.a ${OBOE}/lib/libaccpmcore.a ${OBOE}/lib/libaccpmla.a ${OBOE}/lib/libaccpmoracle.a  ${OBOE}/lib/libaccpmparam.a blas lapack lapackpp)

#mex_setup(MATLAB_ROOT MEX MEX_EXT)
#add_mex(mvilma mvilma_mex.cc OUTPUT_NAME mvilma])

add_executable(main main.cpp ${BMRM} ${ORACLE} ${SOLVER} ${SRC})
target_link_libraries(main ${OBOE_LIBS})

add_executable(report_resuts report_resuts.cpp ${BMRM} ${ORACLE} ${SRC})
target_link_libraries(report_resuts ${OBOE_LIBS})

add_executable(train_vilma train_vilma.cpp ${BMRM} ${ORACLE} ${SRC})
target_link_libraries(train_vilma ${OBOE_LIBS})

add_executable(test_vilma test_vilma.cpp ${BMRM} ${ORACLE} ${SRC})
target_link_libraries(test_vilma ${OBOE_LIBS})

#ADD_MATLAB_MEX_FILE(mvilma mvilma_mex.cc)
# ADD_MATLAB_MEX_FILE(mvilma mvilma_mex.cc sparse_matrix.h sparse_matrix.hpp sparse_vector.h sparse_vector.hpp)
# target_link_libraries(mvilma LINK_PRIVATE ${MEX_LIBRARIES} ${MATLAB_ENGINE_LIBRARIES})

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(MEX_LDFLAGS "-Wl,-rpath")
set(MEX_LINK_LIBS "${OBOE_LIBS};stdc++" CACHE STRING "External libraries to link to this mex file." FORCE)

add_mex(mvilma mvilma_mex.cc ${ORACLE} ${SRC} ${BMRM})
target_link_libraries(mvilma lapack)
target_link_libraries(mvilma blas)
target_link_libraries(mvilma lapackpp)
target_link_libraries(mvilma ${OBOE_LIBS})

# test
add_executable(train_pw_vilma train_pw_vilma.cpp ${BMRM} ${ORACLE} ${SRC})
target_link_libraries(train_pw_vilma ${OBOE_LIBS})

