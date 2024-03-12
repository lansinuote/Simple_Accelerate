#在命令行中运行
#accelerate launch 3.在命令行中运行.py
#accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml 3.在命令行中运行.py --args_for_the_script
#accelerate launch --multi_gpu 3.在命令行中运行.py
#accelerate launch --cpu 3.在命令行中运行.py
#accelerate launch --num_processes=2 3.在命令行中运行.py
#accelerate launch --mixed_precision=fp16 3.在命令行中运行.py
#python -m accelerate.commands.launch 3.在命令行中运行.py

if __name__ == '__main__':
    print('runed')