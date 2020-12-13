export OMP_NUM_THREADS=2
python -m torch.distributed.launch --nproc_per_node=4 run.py configs/mks/mk000.yaml train --dist --dist-val --num-workers 2