FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /work

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["/bin/bash"]
