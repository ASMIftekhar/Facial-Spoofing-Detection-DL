import json
import numpy as np
import csv
import matplotlib.pyplot as plt

def getMetrics(GT, pred):
    FAR = np.sum((GT == 1) & (pred == 0))/len(GT)
    FRR = np.sum((GT == 0) & (pred == 1))/len(GT)
    ACC = np.sum(GT == pred)/len(GT)
    HTER = (FAR + FRR) / 2
    return [FAR, FRR, HTER, ACC]

def tuneHTER(GT, pred):
    for threshold in np.arange(0.01, 1, 0.01):
        preds_bin = np.where(pred > threshold, 1, 0)
        metrics = getMetrics(GT, preds_bin)
        FAR = np.around(metrics[0], decimals=2)
        FRR = np.around(metrics[1], decimals=2)
        #print("Threshold {} | FAR {} | FRR {} | HTER {} | ACC {}".format(threshold, metrics[0], metrics[1], metrics[2],
        #                                                                 metrics[3]))
        eps = 0.01
        if FAR == FRR:
             print("Threshold {} | FAR {} | FRR {} | HTER {} | ACC {}".format(threshold, metrics[0], metrics[1], metrics[2], metrics[3]))
             return threshold
    return 0.5


def getNewMetrics(GT, pred, labels, threshold):
    print_score = 0
    display_score = 0
    real_score = 0

    pred = np.where(pred > threshold, 1, 0)

    n_real = np.sum(np.where(labels == 1, 1, 0))
    n_print = np.sum(np.where(labels == 2, 1, 0) + np.where(labels == 3, 1, 0))
    n_display = np.sum(np.where(labels == 4, 1, 0) + np.where(labels == 5, 1, 0))

    for i in range(GT.shape[0]):
        if GT[i] == 1 and pred[i] == 0: # Need to get FAR for each attack category
            if labels[i] == 2 or labels[i] == 3:
                print_score += 1
            elif labels[i] == 3 or labels[i] == 4:
                display_score += 1
        elif GT[i] == 0 and pred[i] == 1: # And FRR for genuine access
            if labels[i] == 1:
                real_score += 1

    real_score = real_score/n_real
    print_score = print_score/n_print
    display_score = display_score/n_display
    return [real_score, print_score, display_score]




def run_dev(GT_dev,pred_dev,GT,pred,CSV_file,CSV_file_dev):

    labels = []
    with open(CSV_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            codes = line[0].split('_')
            type = codes[-1][0]
            labels.append(int(type))
    labels = np.array(labels)
    import pdb;pdb.set_trace()
  #  GT_dev = np.array(result_dev[0])
  #  pred_dev = np.array(result_dev[1])

    threshold = tuneHTER(np.array(GT_dev),np.array(pred_dev))

    # GT and predictions on test set
  #  GT = np.array(result_test[0])
  #  pred = np.array(result_test[1])

    metrics = getNewMetrics(np.array(GT), np.array(pred), labels, threshold)

    APCER = np.max(metrics[1:])
    final_score = (metrics[0] + APCER) / 2

    print('Final score is -> APCER {} | BPCER {} | ACER {}'.format(APCER, metrics[0], final_score))
    return [APCER, metrics[0], final_score]



if __name__ == "__main__":
    # Get predictions for test and dev set
    protocol = '1'

    CSV_file = 'OULU_Test{}.csv'.format(protocol)
    CSV_file_dev = 'OULU_Dev{}.csv'.format(protocol)

    pred_test = 'oulu_resnet{}\\predictions_test.json'.format(protocol)
    pred_dev = 'oulu_resnet{}\\predictions_dev.json'.format(protocol)
    plot = 'oulu_resnet{}\\plot.json'.format(protocol)
    with open(pred_test) as handle:
        result_test = json.load(handle)
    with open(pred_dev) as handle:
        result_dev = json.load(handle)
    with open(plot) as handle:
        result_plot = json.load(handle)

    # Populate labels with type of attack, normal, print or video
    labels = []
    with open(CSV_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            codes = line[0].split('_')
            type = codes[-1][0]
            labels.append(int(type))
    labels = np.array(labels)

    GT_dev = np.array(result_dev[0])
    pred_dev = np.array(result_dev[1])

    threshold = tuneHTER(GT_dev, pred_dev)
    threshold = 0.75

    # GT and predictions on test set
    GT = np.array(result_test[0])
    pred = np.array(result_test[1])

    metrics = getNewMetrics(GT, pred, labels, threshold)

    APCER = np.max(metrics[1:])
    final_score = (metrics[0] + APCER) / 2

    print('Final score is -> APCER {} | BPCER {} | ACER {}'.format(APCER, metrics[0], final_score))

    cost_train = result_plot[0]
    cost_test = result_plot[1]
    LR = result_plot[2]
    plt.figure()
    plt.plot(range(len(cost_test)), cost_test)
    plt.plot(range(len(cost_test)), cost_train)
    plt.legend(['Test', 'Train'])
    plt.title('Cost for Protocol {}'.format(protocol))
    plt.savefig("COST_prot{}.png".format(protocol))






