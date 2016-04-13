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
    para1 = np.empty(para_len, dtype=float)
    para1.fill(np.nan)
    para2 = np.empty(para_len, dtype=float)
    para2.fill(np.nan)
    para3 = np.empty(para_len, dtype=float)
    para3.fill(np.nan)

    P_l1 = np.empty(para_len, dtype=float)
    P_l1.fill(np.nan)

    P1_l1 = np.empty(para_len, dtype=float)
    P1_l1.fill(np.nan)

    P2_l1 = np.empty(para_len, dtype=float)
    P2_l1.fill(np.nan)

    P3_l1 = np.empty(para_len, dtype=float)
    P3_l1.fill(np.nan)

    for i in range(para_len):
    # for i in range(0, 3):

        row_col_index = id_uniq[i]
        training_mat = training_csr.getrow(row_col_index).toarray()[0, :] if col == 'sender' else \
            training_csr.getcol(row_col_index).toarray()[:, 0]
        # print 'training_mat', training_mat
        # Finding the maximum of the transaction records
        m = np.max(training_mat)

        if m == 0:
            pass
        else:
            m = m if m > 10 else 10
            entries, bin_edges = np.histogram(training_mat, bins=m)  # bins=1000)
            # print 'entries', entries
            # print 'bin_edges', bin_edges
            bin_edges = bin_edges[:-1]

            iteration = 0
            # b0 = float(entries[0])
            # b0 = 0.0
            # b0 = float(entries[0]) * 0.1
            b0 = float(entries[0]) * 0.6
            # b0 = 0.0 if np.sum(entries[1:]) != 0 else float(entries[0])
            # print b0
            while iteration <= 2:

                normalized_entries = entries.astype(float) / (b0 + np.sum(entries[1:]))
                # print 'i', i, 'row_col_index', row_col_index
                # print float(b0)
                # print (b0 + np.sum(entries[1:]))
                # print entries[entries != 0], bin_edges[entries != 0]
                normalized_entries[0] = float(b0) / (b0 + np.sum(entries[1:]))

                # print entries[entries != 0]
                # print normalized_entries
                ### Fitting
                try:
                    lambda_poiss, cov_matrix = curve_fit(poisson, bin_edges, normalized_entries)
                except RuntimeError:
                    lambda_poiss = [np.nan]

                ### Updating b0
                b0_new = (b0 + np.sum(entries[1:])) * poisson(0, lambda_poiss[0])
                b0 = b0_new if entries[0] > b0_new else b0

                if i == 1:
                    print '------ iteration %d ------' % iteration
                    print 'lambda:', lambda_poiss[0]
                    print 'row_col_index:', row_col_index
                    print 'b0:', b0
                    print 'P(l = 1):', (b0 + np.sum(entries[1:])) / float(np.sum(entries))
                    print 'normal entries:', normalized_entries[:10]

                if iteration == 0:
                    para[i] = lambda_poiss[0]
                    P_l1[i] = (b0 + np.sum(entries[1:])) / float(np.sum(entries))
                elif iteration == 1:
                    para1[i] = lambda_poiss[0]
                    P1_l1[i] = (b0 + np.sum(entries[1:])) / float(np.sum(entries))
                elif iteration == 2:
                    para2[i] = lambda_poiss[0]
                    P2_l1[i] = (b0 + np.sum(entries[1:])) / float(np.sum(entries))
                elif iteration == 3:
                    para3[i] = lambda_poiss[0]
                    P3_l1[i] = (b0 + np.sum(entries[1:])) / float(np.sum(entries))

                iteration += 1

                # ### Plotting
                # x_plot = np.linspace(0, 10, 100)
                # plt.figure(num=i)
                # plt.plot(x_plot, poisson(x_plot, *lambda_poiss), 'r-', lw=2)
                # plt.bar(bin_edges, normalized_entries, width=0.4)
                # plt.xlim([0, 10])
                # plt.show()

    return para, para1, para2, para3, P_l1, P1_l1, P2_l1, P3_l1


def method3(para_sender, para_receiver):

    P_s_pois = poisson(0, para_sender)
    P_r_pois = poisson(0, para_receiver)

    prediction_poisson = [(1 - P_s_pois[np.where(testing_df['sender'].unique() == row[1]['sender'])[0][0]]) *
                          (1 - P_r_pois[np.where(testing_df['receiver'].unique() == row[1]['receiver'])[0][0]])
                          for row in testing_df.iterrows()]

    return prediction_poisson


def method4(para_sender, para_receiver, Psl1, Prl1):

    P_s_pois = poisson(0, para_sender)
    P_r_pois = poisson(0, para_receiver)

    prediction_poisson = [(1 - P_s_pois[np.where(testing_df['sender'].unique() == row[1]['sender'])[0][0]]) *
                          Psl1[np.where(testing_df['sender'].unique() == row[1]['sender'])[0][0]] *
                          (1 - P_r_pois[np.where(testing_df['receiver'].unique() == row[1]['receiver'])[0][0]]) *
                          Prl1[np.where(testing_df['receiver'].unique() == row[1]['receiver'])[0][0]]
                          for row in testing_df.iterrows()]

    return prediction_poisson


def plot_roc(prediction, num=1):
    ### Making the ROC curve

    fpr, tpr, thresholds1 = roc_curve(testing_df['transaction'], prediction)
    roc_auc = auc(fpr, tpr)

    print "Area under the ROC curve : %f" % roc_auc

    plt.style.use('ggplot')

    plt.figure(num=num, figsize=(6, 6))

    plt.subplots_adjust(top=0.94, bottom=0.1, right=0.97, left=0.12)

    plt.subplot(111)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)#, color = 'b'
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=17)
    plt.legend(loc="lower right", fontsize=17)
    # plt.show()


def plot_roc3(pred1, pred2, pred3, num=2):
    ### Making the ROC curve

    fpr1, tpr1, thresholds1 = roc_curve(testing_df['transaction'], pred1)
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, thresholds2 = roc_curve(testing_df['transaction'], pred2)
    roc_auc2 = auc(fpr2, tpr2)

    fpr3, tpr3, thresholds3 = roc_curve(testing_df['transaction'], pred3)
    roc_auc3 = auc(fpr3, tpr3)

    print "Area under the ROC curve : %f" % roc_auc1
    print "Area under the ROC curve : %f" % roc_auc2
    print "Area under the ROC curve : %f" % roc_auc3

    plt.style.use('ggplot')

    plt.figure(num=num, figsize=(18, 6))

    ### Layout Settings
    plt.subplots_adjust(top=0.94, bottom=0.105, right=0.985, left=0.045)

    plt.subplot(131)
    plt.plot(fpr1, tpr1, label='ROC curve (area = %0.3f)\nIteration 1' % roc_auc1)#, color='b')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=17)
    plt.legend(loc="lower right", fontsize=17)

    plt.subplot(132)
    plt.plot(fpr2, tpr2, label='ROC curve (area = %0.3f)\nIteration 2' % roc_auc2)#, color='b')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=15)
    # plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=17)
    plt.legend(loc="lower right", fontsize=17)

    plt.subplot(133)
    plt.plot(fpr3, tpr3, label='ROC curve (area = %0.3f)\nIteration 3' % roc_auc3)#, color='b')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=15)
    # plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=17)
    plt.legend(loc="lower right", fontsize=17)
    # plt.show()


def main():

    plot_roc(method1())
    plt.savefig('figures/ROC1.pdf', format='pdf')
    # plot_roc(method2())

    para_sender, para_sender1, para_sender2, para_sender3, Psl1, Ps1l1, Ps2l1, Ps3l1 = get_poisson_para(col='sender')
    para_receiver, para_receiver1, para_receiver2, para_receiver3, Prl1, Pr1l1, Pr2l1, Pr3l1 = get_poisson_para(col='receiver')

    # print 'sum of the difference', np.nansum(para_sender - para_receiver1)

    # plot_roc(method3(para_sender, para_receiver), num=1)
    # plot_roc(method3(para_sender1, para_receiver1), num=2)
    # plot_roc(method3(para_sender2, para_receiver2), num=3)
    # plot_roc(method3(para_sender3, para_receiver3), num=4)
    # plt.show()

    # plot_roc(method4(para_sender, para_receiver, Psl1, Prl1), num=1)
    # plot_roc(method4(para_sender1, para_receiver1, Ps1l1, Pr1l1), num=2)
    # plot_roc(method4(para_sender2, para_receiver2, Ps2l1, Pr2l1), num=3)
    plot_roc3(method4(para_sender, para_receiver, Psl1, Prl1),
              method4(para_sender1, para_receiver1, Ps1l1, Pr1l1),
              method4(para_sender2, para_receiver2, Ps2l1, Pr2l1),
              num=2)
    # plot_roc(method4(para_sender3, para_receiver3, Ps3l1, Pr3l1), num=4)
    plt.savefig('figures/ROC3.pdf', format='pdf')
    # plt.show()

    # para_send = get_poisson_para(col='sender')
    # print para_send
    # print 'length', len(para_send)


if __name__ == '__main__':
    main()
    # get_poisson_para(col='sender')
    # get_poisson_para(col='receiver')
    # para_sender, para_sender1, para_sender2, para_sender3 = get_poisson_para(col='sender')
    # para_receiver, para_receiver1, para_receiver2, para_receiver3 = get_poisson_para(col='receiver')
    # pred = method3(para_sender, para_receiver)
    # print sum(np.isnan(pred))
