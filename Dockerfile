# Use
FROM openeuler/openeuler:latest
RUN echo "sslverify=false" >> /etc/yum.conf \
    && yum -y install python3 python3-pip git \
    && pip3 install torch==2.4.1  transformers==4.45.2  huggingface-hub
ARG outm=/opt/llm/user/compression/output_model
ARG hf_token
RUN if [ -z ${hf_token+x} ]; then echo 'hf_token unspecified. Exiting.'; exit 1; fi
# folder where the model has to be saved. Do not change
ENV OUTPUT_MODEL=${outm}
# Hugginface token. By default, it is retrieved automatically by build.sh. Change that script if necessary
ENV HF_TOKEN=${hf_token}
RUN mkdir -p /opt/llm
WORKDIR /opt/llm
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness.git
#COPY lm-evaluation-harness /opt/llm/lm-evaluation-harness
COPY assets/__init__.py /opt/llm/lm-evaluation-harness/lm_eval/models
COPY assets/user.py /opt/llm/lm-evaluation-harness/lm_eval/models
COPY assets/user /opt/llm/user
COPY assets/run.sh /opt/llm
COPY assets/mean.sh /opt/llm

RUN mkdir -p ${outm}


WORKDIR /opt/llm/lm-evaluation-harness
RUN pip install -e .

WORKDIR /opt/llm
