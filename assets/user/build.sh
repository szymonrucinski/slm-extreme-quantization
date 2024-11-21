#!/bin/bash 
set -x
set -e
# loging to HF
huggingface-cli login --token ${HF_TOKEN}
# install dependencies specified in setup.py
python3 setup.py install
if [[ $1 == "--compress" ]]; then
	for i in {0..10};do echo "";done
	echo "Put here the steps needed to compress the full precision model"
	echo "Any dependencies needed must be specified in setup.py"
	echo "The source files with the compression code need to be stored in compress"
	echo "At the end of this script, all the files needed to LOAD the model for evaluation must be stored in $OUTPUT_MODEL"
	cd compression
	python3 compress.py
fi
echo "BUILD SUCCESSFUL"
