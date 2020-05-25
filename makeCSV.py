import os
import csv
import glob

path = 'C:\\Users\Raphael\\Downloads\\Train_files'

file_names = glob.glob(os.path.join(path, '*.avi'))

with open('OULU.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    for file in file_names:
        words = file.split('\\')
        codes = words[-1].split('_')[-1]
        code = int(codes[0])
        if code == 1:
            label = 'client'
        else:
            label = 'imposter'
        writer.writerow([words[-1], label])




a = 5