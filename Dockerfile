FROM continuumio/miniconda3
RUN conda create -n env python=3.6
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

WORKDIR /code

COPY enviroment.yml /code/envirrroment.yml

RUN conda update -n base -c defaults conda
RUN conda env create -f envirrroment.yml

