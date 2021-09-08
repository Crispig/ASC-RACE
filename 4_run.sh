CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=3 --master_addr="192.168.3.21" --master_port=50000 run.py --data_dir=./dev/shm/RACE --bert_model=./dev/shm/bert-large-uncased.tar.gz --vocab_file=./workspace/model/bert-large-uncased-vocab.txt --output_dir=large_models --max_seq_length=320 --do_train --do_lower_case --train_batch_size=22 --eval_batch_size=12 --learning_rate=2e-4 --num_train_epochs=2 --gradient_accumulation_steps=1 --fp16 --loss_scale=128



