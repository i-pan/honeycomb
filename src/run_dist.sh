export OMP_NUM_THREADS=2
pyt -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk000.yaml train --dist --num-workers 2