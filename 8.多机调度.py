import torch
from functions import get_loader, get_model
import random

#控制随机数
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

_, _, loader = get_loader(text_lens=1)
model, optimizer, scheduler = get_model(num_hidden_layers=4)

from accelerate import Accelerator
import datetime
import os

#机器编号
accelerator = Accelerator()

print('rank=', os.environ.get('RANK', None))
print('local_rank=', os.environ.get('LOCAL_RANK', None))
print('accelerator.distributed_type=', accelerator.distributed_type)
print('accelerator.is_local_main_process=', accelerator.is_local_main_process)
print('accelerator.is_main_process=', accelerator.is_main_process)

loader, model, optimizer, scheduler = accelerator.prepare(
    loader, model, optimizer, scheduler)

now = datetime.datetime.now()
for i, data in enumerate(loader):
    out = model(**data)
    accelerator.backward(out.loss)
    accelerator.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if i % 1 == 0:
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        labels = data['labels']
        logits = out['logits'].argmax(1)
        acc = (labels == logits).sum().item() / len(labels)

        print(i, len(loader), out.loss.item(), lr, acc, accelerator.device)

print(datetime.datetime.now() - now)

accelerator.wait_for_everyone()
if accelerator.is_main_process and accelerator.is_local_main_process:
    print('model.save_pretrained(...)')

#单机运行
#python 8.多机调度.py

#指定要使用的网卡
#export GLOO_SOCKET_IFNAME=enp0s8

#先跑主,再跑从,分布式训练
#python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="192.168.56.104" --master_port=60006 8.多机调度.py --arg1 --arg2
#python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="192.168.56.104" --master_port=60006 8.多机调度.py --arg1 --arg2

#同上
#torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="192.168.56.104" --master_port=60006 8.多机调度.py --arg1 --arg2
#torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="192.168.56.104" --master_port=60006 8.多机调度.py --arg1 --arg2

#同上,不过是accelerate的api,建议直接使用torchrun
#accelerate launch --multi_gpu --num_processes=2 --num_machines=2 --main_process_ip="192.168.56.104" --main_process_port=60006 --machine_rank=0 8.多机调度.py --arg1 --arg2
#accelerate launch --multi_gpu --num_processes=2 --num_machines=2 --main_process_ip="192.168.56.104" --main_process_port=60006 --machine_rank=1 8.多机调度.py --arg1 --arg2
