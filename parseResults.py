import json
import numpy as np

filename = 'predictions.json'

def getMetrics(GT, pred):
    FAR = np.sum((GT == 1) & (pred == 0))/len(GT)
    FRR = np.sum((GT == 0) & (pred == 1))/len(GT)
    ACC = np.sum(GT == pred)/len(GT)
    HTER = (FAR + FRR) / 2
    return [FAR, FRR, HTER, ACC]

def tuneHTER(GT, pred):
    for threshold in np.arange(0.01, 1, 0.01):
        preds_bin = np.where(np.array(pred) > threshold, 1, 0)
        metrics = getMetrics(GT, preds_bin)
        FAR = np.around(metrics[0], decimals=3)
        FRR = np.around(metrics[1], decimals=3)
        #eps = 0.01
        print("Threshold {} | FAR {} | FRR {} | HTER {} | ACC {}".format(threshold, metrics[0], metrics[1], metrics[2], metrics[3]))
        if FAR == FRR:
            print('THIS VALUE')



if __name__ == "__main__":
    with open(filename) as handle:
        result = json.load(handle)

    GT = np.array(result[0])
    pred = result[1]

    tuneHTER(GT, pred)



