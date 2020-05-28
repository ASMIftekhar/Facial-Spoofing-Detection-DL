# ECE283_Project
Download data
```Shell
mkdir dataset
bash download.sh
```
Example generating Train CSV file for protocol 1
```Shell
python3 createCSV2.py -p /media/data/data_spoof/Train_data -pr 1 -t Train
```
Train from scratch 
```Shell
CUDA_VISIBLE_DEVICES=0 python3 main.py -nw 8 -ba 8 -l 1e-5 -fw test
```
Run the best inference 
```Shell
CUDA_VISIBLE_DEVICES=0 python3 main.py -nw 8 -ba 8 -l 1e-5 -fw test -r t -c best -i t
```

