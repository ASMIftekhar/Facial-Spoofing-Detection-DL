import os
import csv
import glob
import argparse

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, default='', help='Path to data')
parser.add_argument('-pr', '--protocol', type=int, required=True, default=1, help='Protocol to use')
parser.add_argument('-t', '--type', type=str, required=True, default='Train', help='Train or test')
args = parser.parse_args()

path = args.path

# Train protocols defs
p1_tr = {'session': [1, 2], 'phone': range(1, 6), 'users': range(1, 20), 'attacks': range(1, 5)}
p2_tr = {'session': [1, 2, 3], 'phone': range(1, 6), 'users': range(36, 55), 'attacks': [2, 4]}
p_tr = [p1_tr, p2_tr]

# Test protocols defs
p1_te = {'session': [3], 'phone': range(1, 6), 'users': range(1, 20), 'attacks': range(1, 5)}
p2_te = {'session': [1, 2, 3], 'phone': range(1, 6), 'users': range(36, 55), 'attacks': [3, 5]}
p_te = [p1_te, p2_te]

if args.type == 'Train':
    p = p_tr[args.protocol]
else:
    p = p_te[args.protocol]

file_names = glob.glob(os.path.join(path, '*.avi'))

output_name = 'OULU_' + args.type + str(args.protocol) + '.csv'

with open(output_name, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    for file in file_names:
        import pdb; pdb.set_trace()
        words = file.split('\\')
        #codes = words[-1].split('_')[-1]
        codes = words[-1].split('_')
        phone = int(codes[0])
        session = int(codes[1])
        user = int(codes[2])
        attack = int(codes[3][0])
        #code = int(codes[0])

        if session in p['session'] and phone in p['phone'] and user in p['users'] and attack in p['attacks']:
            if attack == 1:
                label = 'client'
            else:
                label = 'imposter'
            writer.writerow([words[-1], label])




a = 5
