# Facial Spoofing Detection Using CNN + LSTM

For this project, we developed a pipeline to detect Presentation Attacks using a CNN followed by a LSTM.
We evaluate our approach using the Replay-Attack and OULU spoofing datasets, which are frequently used on the literature.
After experimenting with several architectures as our backbone for feature extraction, we achieved the following results on each dataset using ResNet101:

Replay-Attack:
Accuracy 99.9% and 0.1 Half-Term Error Rate (HTER)

OULU protocol 1:
APCER: 0.4 - BPCER: 4.1 - HTER: 2.29

OULU Protocol 2:
APCER: 0.83 - BPCER: 1.6 - HTER: 1.25


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

