import numpy as np
import matplotlib.pyplot as plt


# todo:check if still necessary
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


def plot_results_v_vs_t(enc_task, path_save, data_dict_v, data_dict_t, size, col_name):  # , maj_cl_val):
    # wasn't able to extract normalized Layer for text-model; temporary fix, drop normalized layer for v_model
    # data_dict_v.drop(data_dict_v.tail(1).index, inplace=True)

    labels = list(data_dict_v[col_name].keys())
    print(labels)
    v_results = data_dict_v[col_name].values
    t_results = data_dict_t[col_name].values
    # dummy_val_t = np.array(data_dict_t['dummy'].values).mean()
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
    """ax.set_xlabel('''      Layers

                  Majority Class:
                  ''' + maj_cl_val)"""
    ax.set_xlabel('Layers')
    ax.set_ylabel('Results Linear Classifier')
    ax.set_title('Visual vs. Text model, Task ' + enc_task + ', Data File ' + col_name.split('.')[0] +
                 ', data-set size ' + str(size))
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.ylim([0.3, 1.0])

    # plt.axhline(y=dummy_val_t, color='r', linestyle='-')

    # plt.show()
    print((path_save + 'v_vs_t_results_' + col_name.split('.')[0] + '_' + str(size) + '.png'))
    plt.savefig(path_save + 'v_vs_t_results_' + col_name.split('.')[0] + '_' + str(size) + '.png')
    plt.close()


# todo:check if substituted by def stack_plots(...):
def sub_plots(enc_task, path_save, data_dict_v, data_dict_t, size):
    save_plot_position = 0
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6))

    for col_v in data_dict_v:
        save_plot_position += 1
        plot_x = None
        plot_y = None

        labels = list(data_dict_v[col_v].keys())
        v_results = data_dict_v[col_v].values
        t_results = data_dict_t[col_v].values

        t_results = np.append(t_results, 0.0)

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        if save_plot_position == 1:
            plot_x = 0
            plot_y = 0
        elif save_plot_position == 2:
            plot_x = 0
            plot_y = 1
        elif save_plot_position == 3:
            plot_x = 1
            plot_y = 0
        elif save_plot_position == 4:
            plot_x = 1
            plot_y = 1

        # fig, ax = plt.subplots()
        rects1 = axs[plot_x, plot_y].bar(x - width / 2, v_results, width, label='Visual Model')
        rects2 = axs[plot_x, plot_y].bar(x + width / 2, t_results, width, label='Text Model')

        axs[plot_x, plot_y].set_xlabel('Layers')
        axs[plot_x, plot_y].set_ylabel('Results Linear Classifier')
        axs[plot_x, plot_y].set_title(col_v.split('.')[0])
        axs[plot_x, plot_y].set_xticks(x, labels)
        axs[plot_x, plot_y].legend()

        axs[plot_x, plot_y].bar_label(rects1, padding=3)
        axs[plot_x, plot_y].bar_label(rects2, padding=3)

        fig.tight_layout()
        axs[plot_x, plot_y].set_ylim([0.3, 1.0])
        # axs[plot_x, plot_y].set_figure(figsize=(1280, 960))

    plt.show()
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_' + str(int(size)) + '.png')
    plt.close()


def stack_plots(enc_task, path_save, data_dict_v_new, data_dict_t_new, data_dict_v_old, data_dict_t_old, config_dict):
    save_plot_position = 0
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6))

    for col_v in sorted(data_dict_v_new.keys(), reverse=True):
        save_plot_position += 1

        labels = list(data_dict_v_new[col_v].keys())

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        plot_x = None
        plot_y = None

        # create position for subplot
        if save_plot_position == 1:
            plot_x = 0
            plot_y = 0
        elif save_plot_position == 2:
            plot_x = 0
            plot_y = 1
        elif save_plot_position == 3:
            plot_x = 1
            plot_y = 0
        elif save_plot_position == 4:
            plot_x = 1
            plot_y = 1

        # new results
        v_results_new = data_dict_v_new[col_v].values
        t_results_new = data_dict_t_new[col_v].values
        t_results_new = np.append(t_results_new, 0.0)

        # old results
        v_results_old = data_dict_v_old['avg'].values
        t_results_old = data_dict_t_old['avg'].values
        t_results_old = np.append(t_results_old, 0.0)

        if config_dict['config'] == 'noise':
            # results difference
            v_results_diff = [element1 - element2 for (element1, element2) in zip(v_results_old, v_results_new)]
            t_results_diff = [element1 - element2 for (element1, element2) in zip(t_results_old, t_results_new)]
            # print('results difference', v_results_diff, t_results_diff)

            axs[plot_x, plot_y].plot(x, v_results_diff, label="Visual Model", linestyle="-", color="royalblue")
            axs[plot_x, plot_y].plot(x, t_results_diff, label="Text Model", linestyle="-", color="darkorange")
            axs[plot_x, plot_y].set_xlabel('Layers')
            axs[plot_x, plot_y].set_ylabel('Difference between Models')
            axs[plot_x, plot_y].set_title('Results Clean vs. Noisy Data, ' + col_v + ', ' + enc_task)
            axs[plot_x, plot_y].set_xticks(x, labels)
            axs[plot_x, plot_y].legend()

            # axs[plot_x, plot_y].fill_between(labels, v_results_new, t_results_new, color="grey", alpha=0.3)

            fig.tight_layout()
            # axs[plot_x, plot_y].set_ylim([0.5, 1.0])

        else:
            axs[plot_x, plot_y].bar(x - width / 2, v_results_old, width, label='Visual Model Old',
                                                 color="lightsteelblue")
            axs[plot_x, plot_y].bar(x + width / 2, t_results_old, width, label='Text Model Old',
                                                 color="bisque")

            axs[plot_x, plot_y].bar(x - width / 2, v_results_new, width*0.5, label='Visual Model New',
                                             color="royalblue", edgecolor='black')
            axs[plot_x, plot_y].bar(x + width / 2, t_results_new, width*0.5, label='Text Model New',
                                             color="darkorange", edgecolor='black')

            axs[plot_x, plot_y].set_xlabel('Layers')
            axs[plot_x, plot_y].set_ylabel('Results Linear Classifier')
            axs[plot_x, plot_y].set_title(col_v)
            axs[plot_x, plot_y].set_xticks(x, labels)
            axs[plot_x, plot_y].legend()

            fig.tight_layout()
            axs[plot_x, plot_y].set_ylim([0.3, 1.0])
        # axs[plot_x, plot_y].set_figure(figsize=(1280, 960))

    plt.show()
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_' + str(int(size)) + '_stack3.png')
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_stack.png')
    plt.close()


def one_plot_old_new(enc_task, path_save, data_dict_v_new, data_dict_t_new, data_dict_v_old, data_dict_t_old, config_dict):
    fig, axs = plt.subplots(figsize=(12.8, 9.6))

    print('data_dict_v_new.keys()', data_dict_v_new.keys())

    for col_v in sorted(data_dict_v_new.keys(), reverse=True):
        print('col_v', col_v)

        labels = list(data_dict_v_new[col_v].keys())

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        # new results
        v_results_new = data_dict_v_new[col_v].values
        t_results_new = data_dict_t_new[col_v].values
        t_results_new = np.append(t_results_new, 0.0)

        # old results
        v_results_old = data_dict_v_old['avg'].values
        t_results_old = data_dict_t_old['avg'].values
        t_results_old = np.append(t_results_old, 0.0)

        if config_dict['config'] == 'noise':
            # results difference
            v_results_diff = [element1 - element2 for (element1, element2) in zip(v_results_old, v_results_new)]
            t_results_diff = [element1 - element2 for (element1, element2) in zip(t_results_old, t_results_new)]
            # print('results difference', v_results_diff, t_results_diff)

            axs.plot(x, v_results_new, label="Visual Model", linestyle="-", color="royalblue")
            axs.plot(x, t_results_new, label="Text Model", linestyle="-", color="darkorange")
            axs.set_xlabel('Layers')
            axs.set_ylabel('Difference between Models')
            axs.set_title('Results Clean vs. Noisy Data, ' + col_v + ', ' + enc_task)
            axs.set_xticks(x, labels)
            axs.legend()

            # axs.fill_between(labels, v_results_new, t_results_new, color="grey", alpha=0.3)

            fig.tight_layout()
            # axs.set_ylim([0.5, 1.0])

        else:
            axs.bar(x - width / 2, v_results_old, width, label='Visual Model Old',
                                                 color="lightsteelblue")
            axs.bar(x + width / 2, t_results_old, width, label='Text Model Old',
                                                 color="bisque")

            axs.bar(x - width / 2, v_results_new, width*0.5, label='Visual Model New',
                                             color="royalblue", edgecolor='black')
            axs.bar(x + width / 2, t_results_new, width*0.5, label='Text Model New',
                                             color="darkorange", edgecolor='black')

            axs.set_xlabel('Layers')
            axs.set_ylabel('Results Linear Classifier')
            axs.set_title(col_v)
            axs.set_xticks(x, labels)
            axs.legend()

            fig.tight_layout()
            axs.set_ylim([0.3, 1.0])
        # axs[plot_x, plot_y].set_figure(figsize=(1280, 960))

    plt.show()
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_' + str(int(size)) + '_stack3.png')
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_stack.png')
    plt.close()
