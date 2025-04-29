FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /work

RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install torchtext
RUN pip install transformers[sentencepiece]

CMD ["/bin/bash"]
