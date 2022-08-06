CODE_DIR=/mnt/d/work/code
MAIN_ROOT=$CODE_DIR/espnet
FAIRSEQ_ROOT=$CODE_DIR/fairseq
S3PRL_ROOT=$CODE_DIR/s3prl
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

export PATH=$PWD/utils/:$PATH
export PATH=$MAIN_ROOT/utils:\
$KALDI_ROOT/tools/openfst/bin:\
$KALDI_ROOT/tools/sph2pipe:\
$MAIN_ROOT/tools/sctk/bin:\
$MAIN_ROOT/espnet/bin:\
$PATH


export LC_ALL=C

# Set PYTHONPATH
export PYTHONPATH=${MAIN_ROOT}:${FAIRSEQ_ROOT}:${S3PRL_ROOT}:${PYTHONPATH:-}

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
