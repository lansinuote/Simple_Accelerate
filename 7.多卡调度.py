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

#使用两张卡
#accelerate launch --num_processes=2 7.多卡调度.py --arg1 --arg2

#所有使用卡
#accelerate launch --multi_gpu 7.多卡调度.py --arg1 --arg2
