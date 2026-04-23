FROM quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11
# FROM quay.io/ascend/python:3.11-ubuntu22.04
# Use the `cann` base image if need to use `bisheng` and run on device
# use `python` base image if just to test binding + `ptoas`

RUN apt update && apt install -y wget git vim

# for on-device execution
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-npu==2.9.0 \
    && pip install --no-cache-dir numpy pyyaml

# extra util
RUN pip install --no-cache-dir \
    pytest pybind11 nanobind setuptools wheel \
    ipython jupyterlab matplotlib pandas

# certain operations need latest isa header, not CANN 8.5.0 default
# header on 2026/04/23
ARG PTOISA_COMMIT=a9613ba881d926fc61f37d12f68658435f3bac9e
WORKDIR /sources
RUN git clone https://gitcode.com/cann/pto-isa.git \
    && cd pto-isa && git checkout $PTOISA_COMMIT

# cache above layers unrelated to ptoas version change

# change this ununsed arg if need to force rebuild later lines
ARG CACHE_BURST=1

# ARG ARCH=x86_64
ARG ARCH=aarch64
ARG RELEASE_REPO=hw-native-sys/PTOAS
ARG RELEASE_VER=0.29
ARG RELEASE_TAG=v${RELEASE_VER}
ARG WHEEL_NAME=ptoas-${RELEASE_VER}-cp311-none-manylinux_2_34_${ARCH}.whl
ARG CLI_TAR_NAME=ptoas-bin-${ARCH}.tar.gz

WORKDIR /installers/

# download assets from https://github.com/hw-native-sys/PTOAS/releases
RUN wget https://github.com/${RELEASE_REPO}/releases/download/nightly/${WHEEL_NAME}
RUN pip install ./${WHEEL_NAME}
RUN python -c "import mlir.ir; from mlir.dialects import pto"

RUN wget https://github.com/${RELEASE_REPO}/releases/download/nightly/${CLI_TAR_NAME}

# set up correct lib path for `ptoas` cli
ARG CLI_DIR=/installers/ptoas-cli
RUN mkdir $CLI_DIR && tar zxf ./${CLI_TAR_NAME} -C $CLI_DIR
RUN chmod +x $CLI_DIR/bin/ptoas
ENV PATH=$CLI_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=$CLI_DIR/lib:$LD_LIBRARY_PATH

# test python binding + cli usage
WORKDIR /tests/
RUN wget https://raw.githubusercontent.com/${RELEASE_REPO}/refs/tags/${RELEASE_TAG}/test/samples/Abs/abs.py
RUN wget https://raw.githubusercontent.com/${RELEASE_REPO}/refs/tags/${RELEASE_TAG}/test/samples/MatMul/tmatmulk.py

RUN python ./tmatmulk.py > ./tmatmulk.pto
RUN ptoas ./tmatmulk.pto -o ./tmatmulk.cpp

RUN python ./abs.py > ./abs.pto
RUN ptoas --enable-insert-sync ./abs.pto -o ./abs.cpp
