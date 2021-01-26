CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
python evaluate.py
python eval_metric.py --pred results/evaluation/ --gt /workspace/fjp/data/VOC2012/SegmentationClassAug/ --test_ids dataset/list/val.txt --save_path dgcn.txt --class_num 21
