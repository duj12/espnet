MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

CODE_DIR=/data/megastore/Projects/DuJing/code
#MAIN_ROOT=$CODE_DIR/espnet
FAIRSEQ_ROOT=$CODE_DIR/fairseq
S3PRL_ROOT=$CODE_DIR/s3prl

#[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
#. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

#. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh

export PYTHONPATH=${MAIN_ROOT}:${FAIRSEQ_ROOT}:${S3PRL_ROOT}:${PYTHONPATH:-}

export PATH=$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe:$KALDI_ROOT/tools/sctk/bin:$PATH
export PATH=$PWD/utils:$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH
export PATH=$MAIN_ROOT/tools/kenlm/build/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

# NOTE(kamo): Source at the last to overwrite the setting
. local/path.sh
