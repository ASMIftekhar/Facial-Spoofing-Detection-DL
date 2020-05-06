# ECE283_Project
Download data
```Shell
mkdir dataset
bash download.sh
```
Train from scratch 
```Shell
CUDA_VISIBLE_DEVICES=0 python3 main.py -nw 8 -ba 8 -l 1e-5 -fw test
```
Run the best inference 
```Shell
CUDA_VISIBLE_DEVICES=0 python3 main.py -nw 8 -ba 8 -l 1e-5 -fw test -r t -c best -i t
```

