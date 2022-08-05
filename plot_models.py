import numpy as np
import matplotlib.pyplot as plt


def plot_results_avg_f_t(enc_task, path_save, m_para, data_dict_all, size):
    a = plt.plot(data_dict_all['avg'].keys(), data_dict_all['avg'].values)
    b = plt.plot(data_dict_all['f_t'].keys(), data_dict_all['f_t'].values)
    plt.xlabel('Layers')
    plt.ylabel('Results Linear Classifier')

    plt.legend(['Averaged tokens', 'First token'])

    if m_para == 'v':
        plt.title('Visual model, task ' + enc_task + ', data-set size ' + str(size))
    else:
        plt.title('Text model, task ' + enc_task + ', data-set size ' + str(size))

    plt.savefig(path_save + m_para + '/' + m_para + '_results_' + enc_task + '_' + str(size) + '.png')
    plt.close()


def plot_results_v_vs_t(enc_task, path_save, data_dict_v, data_dict_t, size, maj_cl_val):
    # wasn't able to extract normalized Layer for text-model; temporary fix, drop normalized layer for v_model
    # data_dict_v.drop(data_dict_v.tail(1).index, inplace=True)

    labels = list(data_dict_v['avg'].keys())
    v_results = data_dict_v['avg'].values
    t_results = data_dict_t['avg'].values
    dummy_val_t = np.array(data_dict_t['dummy'].values).mean()
    # dummy_val_v = np.array(data_dict_v['dummy'].values).mean()

    # adding 0.0 for l7 to text-model data, otherwise I'd need more time to fix this
    t_results = np.append(t_results, 0.0)
    print(t_results)

    print('labels', labels, '\nv_results', v_results, '\nt_results', t_results)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, v_results, width, label='Visual Model')
    rects2 = ax.bar(x + width / 2, t_results, width, label='Text Model')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('''      Layers

                  Majority Class:
                  ''' + maj_cl_val)
    ax.set_ylabel('Results Linear Classifier')
    ax.set_title('Visual vs. Text model, task ' + enc_task + ', data-set size ' + str(size))
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.ylim([0.45, 1.0])

    plt.axhline(y=dummy_val_t, color='r', linestyle='-')

    # plt.show()
    plt.savefig(path_save + 'v_vs_t_results_' + enc_task + '_' + str(size) + '.png')
    plt.close()

