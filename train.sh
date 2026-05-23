export CUDA_DEVICE_ORDER="PCI_BUS_ID"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --nnodes=1 main.py --tasks "SpiralDiff-Canon_EOS_5D" --cfg_path configs/SpiralDIff-Combined.yaml --nGPUs 2 \
