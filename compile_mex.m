clearvars; close all;

eval(['mex -v -O -largeArrayDims mvilma_mex.cc' ' -output ./mvilma_interface']);
