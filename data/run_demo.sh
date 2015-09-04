#!/bin/bash

rm -r demo
mkdir demo

cd demo

featdim=723712
nclasees=55
lambda=0.1
bmrm_cp_buffer_size=300
supervised=1000

ERROR=""

for experiment_id in 0 1000 2000 3000 4000
do
  total=$(($supervised+$experiment_id))
  rm $(pwd)/$total/*
  mkdir $(pwd)/$total
  cd $(pwd)/$total
  cat ../../morph_features.txt | sed -n 1,${total}p > morph_features.txt
  cat ../../morph_supervised_labeling.txt | sed -n 1,${supervised}p > morph_partial_labeling.txt
  offset=$(($supervised+1))
  cat ../../morph_partial_labeling.txt | sed -n ${offset},${total}p >> morph_partial_labeling.txt
  supervised_begin=$((total+1))
  supervised_end=$((total+5000))
  cat ../../morph_supervised_labeling.txt | sed -n ${supervised_begin},${supervised_end}p > morph_supervised_labeling.txt

  ./../../../build/train_vilma $(pwd)/morph_features.txt $(pwd)/morph_partial_labeling.txt $(pwd)/model.bin ${featdim} ${nclasees} ${lambda} ${bmrm_cp_buffer_size}

  ./../../../build/test_vilma $(pwd)/morph_features.txt $(pwd)/morph_supervised_labeling.txt $(pwd)/model.bin ${featdim} ${nclasees} > error.txt
 
  ERROR="$ERROR, $(tail -n1 'error.txt')"
  cd ..
done

cd ..

echo "ERROR:""
echo $ERROR
