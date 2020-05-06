import json
import matplotlib.pyplot as plt
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('-fw','--first_word',type=str,required=False,default='you_dont_put_a_name',help='Name_of_the_file_you_want_to_load')
args=parser.parse_args()
first_word=args.first_word




file_name='results/{}/plot.json'.format(first_word)
with open(file_name) as handle:   
        result = json.load(handle)

train_mean=result[0]
val_mean=result[1]
plt.figure()
plt.plot(train_mean,label='train')
plt.plot(val_mean,label='test')
plt.title('LOG-LOSSES')
plt.legend()
plt.show()
