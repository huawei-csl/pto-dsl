Usage:

```bash
RELEASE_TAG=20260212
sudo docker build . -t pto_dsl:$RELEASE_TAG

# for specific arch (x86_64 vs aarch64)
sudo docker build \
    --build-arg ARCH=x86_64 \
    . -t pto_dsl:$RELEASE_TAG

# to test compile-only
sudo docker run --rm -it pto_dsl:$RELEASE_TAG /bin/bash

# to test on-device execution
sudo docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc  \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $HOME:/mounted_home -w /mounted_home \
    pto_dsl:$RELEASE_TAG /bin/bash
```
