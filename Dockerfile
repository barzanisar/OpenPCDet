# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.8    (apt)
# pytorch       1.6 (pip)
# ==================================================================
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

# ==================================================================
# tools
# ------------------------------------------------------------------
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
	    nano \
        libx11-dev \
        fish \
        libsparsehash-dev \
        software-properties-common \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libssl-dev

# ==================================================================
# python
# ------------------------------------------------------------------
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-distutils \
        python3-apt \
        python3-pip \
        python3-setuptools
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python

# ==================================================================
# conda
# ------------------------------------------------------------------

#RUN mkdir -p /opt/conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH
RUN rm -rf ~/miniconda.sh
ENV PATH /opt/conda/envs/ssl/bin:$PATH

RUN /opt/conda/bin/conda init bash \
    && . ~/.bashrc \
    && conda create -n ssl python=3.8 \
    && conda activate ssl 
RUN echo "source activate ssl" > ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "ssl", "/bin/bash", "-c"]


# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# Install cmake v3.21.3
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://cmake.org/files/v3.21/cmake-3.21.3.tar.gz && \
    tar -xzvf cmake-3.21.3.tar.gz && \
    cd cmake-3.21.3 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp

WORKDIR /root

# Install Boost geometry
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.68.0/source/boost_1_68_0.tar.gz
RUN tar xzvf boost_1_68_0.tar.gz
RUN cp -r ./boost_1_68_0/boost /usr/include
RUN rm -rf ./boost_1_68_0
RUN rm -rf ./boost_1_68_0.tar.gz


# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

#ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;Lovelace"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV MAX_JOBS=4
ENV PYTHONPATH="/usr/lib/python3.8/site-packages/:${PYTHONPATH}"

# ==================================================================
# OpenPCDet
# ------------------------------------------------------------------
WORKDIR /OpenPCDet
#cuda home env needed for minkowski
ENV CUDA_HOME="/usr/local/cuda-11.1" 
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y libgl1

RUN apt-get update -y
RUN apt-get install -y libeigen3-dev
RUN pip install pip==22.1.2
RUN python -m pip --no-cache-dir install -r requirements.txt
RUN conda install openblas-devel -c anaconda
RUN python -m pip --no-cache-dir install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt install libopenblas-dev -y
#RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas" --install-option="--force_cuda"  
RUN pip install nuscenes-devkit

# # Install spconv v1.1
# RUN git clone https://github.com/traveller59/spconv.git
# RUN cd ./spconv && git checkout v1.2.1 && git submodule update --init --recursive && SPCONV_FORCE_BUILD_CUDA=1 python setup.py bdist_wheel
# RUN python -m pip install /root/spconv/dist/spconv*.whl && \
#     rm -rf /root/spconv
# ENV LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/spconv:${LD_LIBRARY_PATH}"

# ==================================================================
# OpenPCDet Framework
# ------------------------------------------------------------------
WORKDIR /OpenPCDet
COPY pcdet pcdet
COPY setup.py setup.py
RUN python setup.py develop
RUN mkdir checkpoints && mkdir data && mkdir output && mkdir tests && mkdir tools && mkdir lib
