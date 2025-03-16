# author: muzhan
# contact: levio.pku@gmail.com
import os
import sys
import time


def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory


def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 5
    cmd = 'CUDA_VISIBLE_DEVICES=' + str(i) + ' python run_exp7.py --num-training-epochs 450 --batch-size 32 --num-data-loaders 56 --save-path backup/exp7-resnet18 --use-tensorboard'
    while gpu_memory > 1000 or gpu_power > 200:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        #i = i % 6
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        #i += 5
    print('\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()