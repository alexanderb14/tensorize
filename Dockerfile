FROM ubuntu:22.04
ENV TZ="Europe/London"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN echo $TZ > /etc/timezone && rm -rf /var/lib/apt/lists/*

RUN apt update -y

# Install essentials
RUN apt install -y software-properties-common cmake git clang clang-format wget vim

# Configure git
RUN git config --global push.default simple
RUN git config --global user.name user
RUN git config --global user.email user@mail.com

# Install Python 3
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update -y
RUN apt install -y python3.10 python3.10-venv python3.10-dev python3-pip

# Install R and R packages
RUN apt install -y r-base
RUN Rscript -e "install.packages('tidyverse')"
RUN Rscript -e "install.packages('scales')"
RUN Rscript -e "install.packages('cowplot')"
RUN Rscript -e "install.packages('RColorBrewer')"
RUN Rscript -e "install.packages('reshape2')"

# Install build dependencies
RUN apt install -y ninja-build bison flex libtool jq lld libgmp3-dev build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libboost-all-dev default-jdk maven

# Set up environment
WORKDIR /root/
COPY . .

# Build dependencies
WORKDIR /root/
RUN cd ${HOME}
RUN python3.10 -m pip install numpy pybind11 toml pyparsing pymlir
RUN ./build_tools/build_dependencies.sh
RUN python3.10 -m pip install -r deps/llvm-project/mlir/python/requirements.txt

# Build Tensorize
RUN ./build_tools/build.sh
RUN python3.10 -m pip install -r requirements.txt
RUN ln -s /usr/bin/python3.10 /usr/bin/python

ENV PYTHONPATH="/root/build/python_packages/synth"

ENTRYPOINT ["/bin/bash", "-l", "-c"]