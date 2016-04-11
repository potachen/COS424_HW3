#!/usr/local/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit
from scipy.misc import factorial
from sklearn.metrics import roc_curve, auc

import time


### --- Tool function block --------------------------------------------------------- ###
### Decorator
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        output = func(*args, **kwargs)
        t2 = time.time()
        print 'Program Running Time: %.8f seconds.' % (t2 - t1)
        return output
    return wrapper


def poisson(x, l):
    return (l**x / factorial(x)) * np.exp(-l)
### --- Tool function block --------------------------------------------------------- ###


### Loading data
training_df = pd.read_csv('data/txTripletsCounts.txt', header=None, index_col=None, sep=' ',
                          names=['sender', 'receiver', 'transaction'])

testing_df = pd.read_csv('data/testTriplets.txt', header=None, index_col=None, sep=' ',
                         names=['sender', 'receiver', 'transaction'])

### Calculating the dimension of the matrix
dim = max((training_df['sender'].max(),
           training_df['receiver'].max(),
           testing_df['sender'].max(),
           testing_df['receiver'].max()))
### Adding 1 because the index starts from 0
dim += 1

### Creating the Compressed Sparse Row matrix
training_csr = csr_matrix((training_df['transaction'], (training_df['sender'], training_df['receiver'])),
                          shape=(dim, dim), dtype=float)


def method1():
    ### Getting Ps and Pr
    row_sum = training_csr.sum(0)
    col_sum = training_csr.sum(1)
    total_sum = training_csr.sum()

    Ps_one = row_sum / float(total_sum)
    Pr_one = col_sum / float(total_sum)

    ### Making prediction list
    prediction1 = [Ps_one[0, row[1]['sender']] * Pr_one[row[1]['receiver'], 0] for row in testing_df.iterrows()]

    return prediction1


def method2():
    ### Getting Ps and Pr
    row_count = (training_csr != 0).sum(0)
    col_count = (training_csr != 0).sum(1)
    total_count = (training_csr != 0).sum()

    Ps_two = row_count / float(total_count)
    Pr_two = col_count / float(total_count)

    ### Making prediction list
    prediction2 = [Ps_two[0, row[1]['sender']] * Pr_two[row[1]['receiver'], 0] for row in testing_df.iterrows()]

    return prediction2


@timing_decorator
def get_poisson_para(col='sender'):
    """ Getting Poisson parameter for 'sender' or 'receiver' """

    ### Initializing parameter list
    id_uniq = testing_df[col].unique()
    para_len = len(id_uniq)
    para = np.empty(para_len, dtype=float)
    para.fill(np.nan)

    para1 = para[:]
    para2 = para[:]
    para3 = para[:]

    for i in range(para_len):
    # for i in range(0, 1):

        row_index = id_uniq[i]
        training_mat = training_csr.getrow(row_index).toarray()
        # Finding the maximum of the transaction records
        m = np.max(training_mat)

        if m == 0:
            pass
        else:
            entries, bin_edges = np.histogram(training_mat, bins=m)  # bins=1000)
            print bin_edges
            bin_edges = bin_edges[:-1]

            iteration = 0
            # b0 = float(entries[0])
            b0 = 0.0
            # b0 = 0.0 if np.sum(entries[1:]) != 0 else float(entries[0])
            # print b0
            while iteration <= 3:

                normalized_entries = entries.astype(float) / (b0 + np.sum(entries[1:]))
                print 'i', i, 'row_index', row_index
                print float(b0)
                print (b0 + np.sum(entries[1:]))
                print entries[entries != 0], bin_edges[entries != 0]
                normalized_entries[0] = float(b0) / (b0 + np.sum(entries[1:]))

                # print entries[entries != 0]
                # print normalized_entries
                ### Fitting
                try:
                    lambda_poiss, cov_matrix = curve_fit(poisson, bin_edges, normalized_entries)
                except RuntimeError:
                    lambda_poiss = np.nan

                ### Updating b0
                b0_new = (b0 + np.sum(entries[1:])) * poisson(0, lambda_poiss)
                b0 = b0_new if entries[0] > b0_new else b0

                if i == 1:
                    print b0

                if iteration == 0:
                    para[i] = lambda_poiss
                elif iteration == 1:
                    para1[i] = lambda_poiss
                elif iteration == 2:
                    para2[i] = lambda_poiss
                elif iteration == 3:
                    para3[i] = lambda_poiss

                iteration += 1

                # ### Plotting
                # x_plot = np.linspace(0, 10, 100)
                # plt.figure(num=i)
                # plt.plot(x_plot, poisson(x_plot, *lambda_poiss), 'r-', lw=2)
                # plt.bar(bin_edges, normalized_entries, width=0.4)
                # plt.xlim([0, 10])
                # plt.show()

    return para, para1, para2, para3


def method3(para_sender, para_receiver):

    P_s_pois = poisson(0, para_sender)
    P_r_pois = poisson(0, para_receiver)

    prediction_poisson = [(1 - P_s_pois[np.where(testing_df['sender'].unique() == row[1]['sender'])[0][0]]) *
                          (1 - P_r_pois[np.where(testing_df['receiver'].unique() == row[1]['receiver'])[0][0]])
                          for row in testing_df.iterrows()]

    return prediction_poisson


def plot_roc(prediction):
    ### Making the ROC curve

    fpr, tpr, thresholds1 = roc_curve(testing_df['transaction'], prediction)
    roc_auc = auc(fpr, tpr)

    print "Area under the ROC curve : %f" % roc_auc

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='b', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()


def main():

    # plot_roc(method1())
    # plot_roc(method2())

    para_sender, para_sender1, para_sender2, para_sender3 = get_poisson_para('sender')
    para_receiver, para_receiver1, para_receiver2, para_receiver3 = get_poisson_para('receiver')

    plot_roc(method3(para_sender, para_receiver))
    plot_roc(method3(para_sender1, para_receiver1))
    plot_roc(method3(para_sender2, para_receiver2))
    plot_roc(method3(para_sender3, para_receiver3))
    plt.show()

    # para_send = get_poisson_para(col='sender')
    # print para_send
    # print 'length', len(para_send)


if __name__ == '__main__':
    main()

