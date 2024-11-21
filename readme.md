# Docker requirements
The development environment is provided through a docker image. The docker container that is built from this image requires access to the GPU to run CUDA code. To meet this requirement, the user must install docker and the nvidia-container-toolkit to use the nvidia runtime.

Please refer to  https://docs.docker.com/engine/install/ for instructions on how to install docker and to https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html for instructions on how to install the nvidia runtime for docker.

# Building and running the image
Build the docker container by running
`./build.sh`
The `build.sh` script also passes to the huggingface token located in `${HOME}/.cache/huggingface/token`.  If the token to be used resides in a different location, please update the `build.sh` script accordingly.
The `build.sh` script creates a  new image with tag `huawei-llm:0.1`.
Create a container with this image by running
`./exec.sh`
This command also logs you on the machine.

# Coding and running the model
Once in the container, the user is required to code and build their solution while meeting these criteria.
1. The user can define arbitrary code in /opt/llm/user to compress the Llama-3.1-8B model and run the compressed model. 
2. The script /opt/llm/user/build.sh is the entry point to install any dependencies, and to compress the model. The scripts must install all the dependencies needed by the user code (see also setup.py). When invoked with the --compress flag, the script must generate the compressed model, that has to be stored in the folder pointed by the env variable ${OUTPUT_MODEL}. The entry point for the compression algorithm must be the "compression/compress.py" file, which is called by the script.
3. All the files needed to load the model must be stored in ${OUTPUT_MODEL}. The size of the content of this folder is used to measure the size of the model. An example of how to store a pytorch model as a file is provided in `user_model.py`.
3. All the dependencies for the model must be installed using the `build.sh` script that is in `/opt/llm/users/`.  This script also needs to be called after each change to the code to make it visible to the `lm-evaluation-harness` benchmark. 
4. Once the model is compressed, it must be used in the file `lm-evaluation-harness/lm_eval/models/user.py`, which must expose the default APIs for evaluation in the framework (https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)
5. The model can be evaluated locally using the `run.sh` script, that computes the average accuracy of the model on selected tasks, the size of the model, and the score assigned to the model. The score is simply the ratio between the accuracy of the model and its size (higher is better). The user can try different datasets among the ones available in the framework.

The user can try the stub provided as an example by running
`bash build.sh --compress` and then ` bash run.sh`

# Submission files
The user must submit a zip file that contains the whole 'user' folder, with the compressed model in ${OUTPUT_MODEL}, and all the files/dependencies needed to generate and invoke the model. The user also needs to submit the `user.py` file in `lm-evaluation-harness/lm_eval/models`. 
The zip file, hence must look like this
```
file.zip
|
|- user
|- user.py
```
