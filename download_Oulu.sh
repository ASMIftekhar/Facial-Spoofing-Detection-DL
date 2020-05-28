
wget --user=Oulu_NPU --password='6!0metr1c5' --progress=bar:force 'http://www.ee.oulu.fi/research/bmetric/Oulu_NPU/Protocols.tar' -O tmp.tar 
tar xvf tmp.tar Protocols/ && rm tmp.tar
wget --user=Oulu_NPU --password='6!0metr1c5' --progress=bar:force 'http://www.ee.oulu.fi/research/bmetric/Oulu_NPU/Train_files.tar' -O tmp.tar
tar xvf tmp.tar Train_files/ && rm tmp.tar
wget --user=Oulu_NPU --password='6!0metr1c5' --progress=bar:force 'http://www.ee.oulu.fi/research/bmetric/Oulu_NPU/Test_files.tar' -O tmp.tar
tar xvf tmp.tar Test_files/ && rm tmp.tar
wget --user=Oulu_NPU --password='6!0metr1c5' --progress=bar:force 'http://www.ee.oulu.fi/research/bmetric/Oulu_NPU/Dev_files.tar' -O tmp.tar
tar xvf tmp.tar Dev_files/ && rm tmp.tar
wget --user=Oulu_NPU --password='6!0metr1c5' --progress=bar:force 'http://www.ee.oulu.fi/research/bmetric/Oulu_NPU/Baseline.tar' -O tmp.tar
tar xvf tmp.tar Baseline/ && rm tmp.tar
wget --user=Oulu_NPU --password='6!0metr1c5' --progress=bar:force 'http://www.ee.oulu.fi/research/bmetric/Oulu_NPU/Readme.pdf' -O Readme.pdf

