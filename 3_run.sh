CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=3 --node_rank=0 --master_addr="115.24.15.47" --master_port=50000 run.py --data_dir=./RACE --bert_model=./dev/shm/bert-large-uncased.tar.gz --vocab_file=./workspace/model/bert-large-uncased-vocab.txt --output_dir=large_models --max_seq_length=320 --do_train --do_lower_case --train_batch_size=20 --eval_batch_size=12 --learning_rate=2e-4 --num_train_epochs=2 --gradient_accumulation_steps=1 --fp16 --loss_scale=128



