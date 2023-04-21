FROM graphcore/poplar:3.2.0-ubuntu-20.04-20230314 as TRITON

RUN apt-get update && apt-get install libssl1.1 -y
#COPY 9395166987522538138.popef /models/unet/1/executable.popef
COPY 14071415351587102118.popef /models/unet/1/executable.popef
COPY config.pbtxt /models/unet/

FROM ubuntu:20.04 as builder
LABEL Graphcore Ltd

#Install the build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    git \
    python3 \
    python3-pip \
    python3-distutils \
    make \
    g++ \
    gcc \
    cmake \
    libssl-dev \
    zlib1g-dev \
    rapidjson-dev \
    libboost-dev \
    libre2-dev \
    librdmacm-dev

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install pip --upgrade
RUN pip3 install cmake

#Build triton backend
ENV TAG r21.05
RUN git clone http://github.com/triton-inference-server/server.git && \
    cd server && \
    git checkout $TAG && \
    mkdir -p mybuild/tritonserver/install && \
    python3 build.py \
        --build-dir mybuild \
        --no-container-build \
        --endpoint=grpc \
        --enable-logging \
        --enable-stats \
        --cmake-dir `pwd`/build \
        --repo-tag=common:$TAG \
        --repo-tag=core:$TAG \
        --repo-tag=backend:$TAG \
        --repo-tag=thirdparty:$TAG

FROM TRITON

COPY --from=builder /server/mybuild/tritonserver/build/server/mybuild/tritonserver/install/bin/ /usr/bin/
COPY --from=builder /server/mybuild/tritonserver/build/server/mybuild/tritonserver/install/lib/ /usr/lib/x86_64-linux-gnu/
COPY --from=builder /server/mybuild/tritonserver/build/server/mybuild/tritonserver/install/include / /usr/include/
RUN mkdir -p /opt/triton/lib/poplar && rm -rf /opt/triton/lib/libtriton_poplar.so
COPY libtriton_poplar.so /opt/triton/lib/poplar

ENV PATH $PATH:/server/mybuild/tritonserver/build/server/mybuild/tritonserver/install/bin

chengshunx@graphcore-5468m6:/localdata/chengshunx/stable_diffusion/save$
chengshunx@graphcore-5468m6:/localdata/chengshunx/stable_diffusion/save$
chengshunx@graphcore-5468m6:/localdata/chengshunx/stable_diffusion/save$ cat Dockerfile
FROM graphcore/poplar:3.2.0-ubuntu-20.04-20230314 as TRITON

RUN apt-get update && apt-get install libssl1.1 -y
#COPY 9395166987522538138.popef /models/unet/1/executable.popef
COPY 14071415351587102118.popef /models/unet/1/executable.popef
COPY config.pbtxt /models/unet/

FROM ubuntu:20.04 as builder
LABEL Graphcore Ltd

#Install the build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    git \
    python3 \
    python3-pip \
    python3-distutils \
    make \
    g++ \
    gcc \
    cmake \
    libssl-dev \
    zlib1g-dev \
    rapidjson-dev \
    libboost-dev \
    libre2-dev \
    librdmacm-dev

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install pip --upgrade
RUN pip3 install cmake

#Build triton backend
ENV TAG r21.05
RUN git clone http://github.com/triton-inference-server/server.git && \
    cd server && \
    git checkout $TAG && \
    mkdir -p mybuild/tritonserver/install && \
    python3 build.py \
        --build-dir mybuild \
        --no-container-build \
        --endpoint=grpc \
        --enable-logging \
        --enable-stats \
        --cmake-dir `pwd`/build \
        --repo-tag=common:$TAG \
        --repo-tag=core:$TAG \
        --repo-tag=backend:$TAG \
        --repo-tag=thirdparty:$TAG

FROM TRITON

COPY --from=builder /server/mybuild/tritonserver/build/server/mybuild/tritonserver/install/bin/ /usr/bin/
COPY --from=builder /server/mybuild/tritonserver/build/server/mybuild/tritonserver/install/lib/ /usr/lib/x86_64-linux-gnu/
COPY --from=builder /server/mybuild/tritonserver/build/server/mybuild/tritonserver/install/include / /usr/include/
RUN mkdir -p /opt/triton/lib/poplar && rm -rf /opt/triton/lib/libtriton_poplar.so
COPY libtriton_poplar.so /opt/triton/lib/poplar

ENV PATH $PATH:/server/mybuild/tritonserver/build/server/mybuild/tritonserver/install/bin

