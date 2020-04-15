import numpy as np
import numpy

import torch

if torch.__version__ != '1.1.0':
    raise RuntimeError('PyTorch version must be 1.1.0')
from torch.nn.functional import softmax


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.conf_matrix = np.zeros((num_classes, num_classes), int)

    def update_matrix(self, out, target):
        for j in range(len(target)):
            self.conf_matrix[out[j].item(), target[j].item()] += 1


    def get_metrics(self):
        samples_for_class = np.sum(self.conf_matrix, 0)
        diag = np.diagonal(self.conf_matrix)

        acc = np.sum(diag) / np.sum(samples_for_class)
        w_acc = np.divide(diag, samples_for_class)
        w_acc = np.mean(w_acc)

        return acc, w_acc


'''
#################
#################
Calibration Utils
#################
#################
'''


# expected calibration error
def compute_ECE(acc_bin, conf_bin, samples_per_bin):
    '''
    Computes expected calibration error:
      acc_bin = array (torch or numpy) where each position is the accuracy of a bin
      conf_bin = array (torch or numpy) where each position is the average confidence of a bin
      samples_per_bin = array (torch or numpy) number of samples per bin
    '''
    assert len(acc_bin) == len(conf_bin)
    ece = 0.0
    total_samples = float(samples_per_bin.sum())
    ece_list = []
    for samples, acc, conf in zip(samples_per_bin, acc_bin, conf_bin):
        ece_list.append(samples / total_samples * numpy.abs(acc - conf))
        ece += samples / total_samples * numpy.abs(acc - conf)
    return ece, ece_list


# maximum calibration error
def compute_MCE(acc_bin, conf_bin, samples_per_bin):
    '''
    Computes Maximum Calibration Error:
       inputs: see compute_ECE function
    '''
    assert len(acc_bin) == len(conf_bin)
    mce = 0.0
    sample = 0
    total_samples = float(samples_per_bin.sum())
    for i in range(len(samples_per_bin)):
        a = samples_per_bin[i] / total_samples * numpy.abs(acc_bin[i] - conf_bin[i])
        if a > mce:
            mce = a
            sample = i
    return mce, sample


# accuracy per bin
def accuracy_per_bin(predicted, real_tag, n_bins, apply_softmax):
    '''
    Computes the accuracy per bin. Each bin represents a partition of the probability space (0-1)
    predicted -> predictions of our model (batch,network_output)
    real_tag -> vector with categorical labels
    n_bins -> number of bins in which the particion space is dividided
    apply_softmax -> apply_softmax to the predictions
    '''

    if apply_softmax:
        predicted_prob = softmax(predicted, dim=1).data
    else:
        predicted_prob = predicted.data

    max_confidence, index = torch.max(predicted_prob, 1)
    correct_label = index.long() == real_tag

    prob = numpy.linspace(0, 1, n_bins + 1)
    acc = numpy.linspace(0, 1, n_bins + 1)
    total_data = len(max_confidence)
    samples_per_bin = []
    for p in range(len(prob) - 1):
        # find elements with probability in between p and p+1
        min_ = prob[p]
        max_ = prob[p + 1]
        boolean_upper = max_confidence <= max_

        if p == 0:  # we include the first element in bin
            boolean_down = max_confidence >= min_
        else:  # after that we included in the previous bin
            boolean_down = max_confidence > min_

        index_range = boolean_down & boolean_upper
        label_sel = correct_label[index_range]

        if len(label_sel) == 0:
            acc[p] = 0.0
        else:
            acc[p] = label_sel.sum().float() / float(len(label_sel))

        samples_per_bin.append(len(label_sel))

    samples_per_bin = numpy.array(samples_per_bin)
    acc = acc[0:-1]
    prob = prob[0:-1]
    return acc, prob, samples_per_bin


# aaverage confidence per bin per bin
def average_confidence_per_bin(predicted, n_bins, apply_softmax):
    '''
    Computes the average confidence per bin. Each bin represents a partition of the probability space (0-1)
    predicted -> predictions of our model (batch,network_output)
    n_bins -> number of bins in which the particion space is dividided
    apply_softmax -> apply_softmax to the predictions
    '''

    if apply_softmax:
        predicted_prob = softmax(predicted, dim=1).data
    else:
        predicted_prob = predicted.data

    prob = numpy.linspace(0, 1, n_bins + 1)
    conf = numpy.linspace(0, 1, n_bins + 1)
    max_confidence, index = torch.max(predicted_prob, 1)

    samples_per_bin = []

    for p in range(len(prob) - 1):
        # find elements with probability in between p and p+1
        min_ = prob[p]
        max_ = prob[p + 1]
        boolean_upper = max_confidence <= max_

        if p == 0:  # we include the first element in bin
            boolean_down = max_confidence >= min_
        else:  # after that we included in the previous bin
            boolean_down = max_confidence > min_

        index_range = boolean_down & boolean_upper
        prob_sel = max_confidence[index_range]

        if len(prob_sel) == 0:
            conf[p] = 0.0
        else:
            conf[p] = prob_sel.sum().float() / float(len(prob_sel))

        samples_per_bin.append(len(prob_sel))

    samples_per_bin = numpy.array(samples_per_bin)
    conf = conf[0:-1]
    prob = prob[0:-1]

    return conf, prob, samples_per_bin


# brier score
def compute_brier(prob, true_labels):
    '''
    computes the brier score
      prob-> vector of probabilities assigned to each class
      acc -> true lables in one-hot format
    '''
    return torch.pow(prob - true_labels, 2).mean()


# computes ECE, MCE, BRIER score and Negative Log Likelihood
def compute_calibration_measures(predictions, true_labels, apply_softmax, bins):
    '''
    Computes several calibration measures
     predictions-> output of your probabilistic model
     true_labels -> ground truth label in categorical format
     apply_softmax -> (true/false)
     bins -> number of bins to partition the probabilistic space
    '''

    predictions = softmax(predictions, 1) if apply_softmax else predictions

    ''' ECE and MCE'''
    acc_bin, prob, samples_per_bin = accuracy_per_bin(predictions, true_labels, n_bins=bins,
                                                      apply_softmax=apply_softmax)
    conf_bin, prob, samples_per_bin = average_confidence_per_bin(predictions, n_bins=bins, apply_softmax=apply_softmax)
    ECE, _ = compute_ECE(acc_bin, conf_bin, samples_per_bin)
    MCE, _ = compute_MCE(acc_bin, conf_bin, samples_per_bin)

    '''Brier Score'''
    max_val = predictions.size(1)
    t_one_hot = categorical_to_one_hot(true_labels, max_val)
    BRIER = compute_brier(predictions, t_one_hot)

    ''' NNL '''
    nnl_eps = 1e-15
    NNL = ((t_one_hot * (-1 * torch.log(predictions + nnl_eps))).sum(1)).mean()

    return ECE, MCE, BRIER, NNL


'''
#####################
END Calibration Utils
#####################
'''


# compute one_hot format lables
def categorical_to_one_hot(t, max_val):
    one_hot = torch.zeros(t.size(0), max_val)
    one_hot.scatter_(1, t.view(-1, 1), 1)
    return one_hot


'''
Differential Entropy
'''


def entropy_categorical(p):
    '''
    Differential entropy of categorical distribution
        input: matrix of shape (batch,probs) where probs is the probability for each of the outcomes
        output: matrix of shape (batch,) with the entropy per each value
    '''
    return (-1 * p * torch.log(p + 1e-34)).sum(-1)
