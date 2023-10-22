import sys
import seaborn as sns
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
        print('x',x)
        x_v = list(range(1, len(labels)+1))
        x_t = list(range(1, len(labels)))
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

        # old results
        v_results_old = data_dict_v_old['avg'].values
        t_results_old = data_dict_t_old['avg'].values

        # bar plots need same length
        if config_dict['config'] != 'noise':
            t_results_new = np.append(t_results_new, 0.0)
            t_results_old = np.append(t_results_old, 0.0)

        if config_dict['config'] == 'noise':
            # results difference
            v_results_diff = [element1 - element2 for (element1, element2) in zip(v_results_old, v_results_new)]
            t_results_diff = [element1 - element2 for (element1, element2) in zip(t_results_old, t_results_new)]
            # print('results difference', v_results_diff, t_results_diff)

            axs[plot_x, plot_y].plot(x_v, v_results_diff, label="Visual Model", linestyle="-", color="royalblue")
            axs[plot_x, plot_y].plot(x_t, t_results_diff, label="Text Model", linestyle="-", color="darkorange")
            axs[plot_x, plot_y].set_xlabel('Layers')
            axs[plot_x, plot_y].set_ylabel('Difference between Models')
            axs[plot_x, plot_y].set_title('Results Clean vs. Noisy Data, ' + col_v + ', ' + enc_task)
            # axs[plot_x, plot_y].set_xticks(x, labels)
            axs[plot_x, plot_y].legend()

            # axs[plot_x, plot_y].fill_between(labels, v_results_new, t_results_new, color="grey", alpha=0.3)

            fig.tight_layout()
            axs[plot_x, plot_y].set_ylim([-0.0025, 0.16])

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
    plt.savefig(path_save + enc_task + '_v_vs_t_results_stack.png')
    plt.close()


def line_plot_prob_tasks_v1(file_v, file_t, classifier, path_save):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 9.6))

    # Set the axes title font size
    plt.rc('axes', titlesize=22)  # Set the axes labels font size
    # Set the font size of the figure title
    plt.rc('figure', titlesize=25)

    x_v_axis = file_v.index.to_list()
    x_t_axis = file_t.index.to_list()

    colors_list = sns.color_palette('Paired')

    # colors_list = ["palegreen", "hotpink", "cornflowerblue", "golden"]
    colors_list = ['#f9d030', '#b8ee30', '#26dfd0', '#f62aa0']
    # colors_list = ["royalblue", "darkorange", "green", "red", "gold", "turquoise", "magenta"]
    dot_symbols_list = ['o', 'v', '*', 'H', '^', 'x', 'D']

    prob_tasks_list = sorted(file_v.keys())

    # for task, color, symbol in zip(prob_tasks_list, colors_list, dot_symbols_list):
    for i in range(len(prob_tasks_list)):
        # print('layer ', layer[-1], '\n color ', color, '\n symbol ', symbol)
        # print(i*2, i*2-1, i*3-1)
        ax1.plot(x_v_axis, file_v[prob_tasks_list[i]], linestyle='-', color=colors_list[i], marker=dot_symbols_list[i], markersize=15)
        ax2.plot(x_t_axis, file_t[prob_tasks_list[i]], linestyle='-', color=colors_list[i], marker=dot_symbols_list[i], markersize=15)

    ax1.legend(prob_tasks_list, loc='lower right', fontsize="18")
    ax2.legend(prob_tasks_list, loc='lower right', fontsize="18")

    ax1.set_xlabel('Layers', fontsize=18)
    ax2.set_xlabel('Layers', fontsize=18)

    ax1.set_ylabel('Probing Accuracy', fontsize=18)
    ax2.set_ylabel('Probing Accuracy', fontsize=18)

    ax1.set_title('Visual Encoder')
    ax2.set_title('Text Encoder')

    if classifier == 'lr':
        fig.suptitle('Probing Tasks Results for Linear Regression Classifier')
        # fig.suptitle('F1-Scores for Linear Regression Classifier')
        # fig.suptitle('Normalized F1-Scores for Linear Regression Classifier')
        ax1.set_ylim([0.55, .97])
        ax2.set_ylim([0.55, .97])

        ax1.set_yticks(np.arange(0.55, .97, step=0.05))
        ax2.set_yticks(np.arange(0.55, .97, step=0.05))
    elif classifier == 'mlp':
        fig.suptitle('Probing Tasks Results for Multi-Layer-Perceptron Classifier')
        # fig.suptitle('F1-Scores for Multi-Layer-Perceptron Classifier')
        # fig.suptitle('Normalized F1-Scores for Multi-Layer-Perceptron Classifier')
        ax1.set_ylim([0.6, .97])
        ax2.set_ylim([0.6, .97])

        ax1.set_yticks(np.arange(0.6, .97, step=0.05))
        ax2.set_yticks(np.arange(0.6, .97, step=0.05))

    # ax1.set_ylim([0.05, 1.05])
    # ax2.set_ylim([0.05, 1.05])

    # ax1.set_yticks(np.arange(0.05, 1.05, step=0.05))
    # ax2.set_yticks(np.arange(0.05, 1.05, step=0.05))



    fig.tight_layout()
    # plt.show()

    plt.savefig(path_save + 'prob_tasks_results_10kit.png')
    print(path_save + 'prob_tasks_results_10kit.png')


def line_plot_prob_tasks_v2(file_v, file_t, classifier, path_save):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12.8, 9.6))

    # Set the axes title font size
    plt.rc('axes', titlesize=22)  # Set the axes labels font size
    # Set the font size of the figure title
    plt.rc('figure', titlesize=25)

    ax_list = [ax1[0], ax1[1], ax2[0], ax2[1]]

    x_v_axis = file_v.index.to_list()
    x_t_axis = file_t.index.to_list()

    # colors_list = ["royalblue", "darkorange"]  #, "green", "red", "gold", "turquoise", "magenta"]
    colors_list = sns.color_palette('colorblind', 6)
    dot_symbols_list = ['o', 'v']  # , '*', 'x', '^', 'H', 'D']

    prob_tasks_list = sorted(file_v.keys())

    for ax_elem, prob_task in zip(ax_list, prob_tasks_list):
        ax_elem.plot(x_v_axis, file_v[prob_task], linestyle='-', color=colors_list[0], marker='o', markersize=15)
        ax_elem.plot(x_t_axis, file_t[prob_task], linestyle='-', color=colors_list[1], marker='v', markersize=15)

        ax_elem.legend(['Visual Model', 'Text Model'], loc='lower right', fontsize="18")
        ax_elem.set_xlabel('Layers', fontsize=18)
        ax_elem.set_ylabel('Probing Accuracy', fontsize=18)
        ax_elem.set_title(prob_task)

    if classifier == 'lr':
        fig.suptitle('Probing Tasks Results for Linear Regression Classifier')
        # fig.suptitle('F1-Scores for Linear Regression Classifier')
    elif classifier == 'mlp':
        fig.suptitle('Probing Tasks Results for Multi-Layer-Perceptron Classifier')
        # fig.suptitle('F1-Scores for Multi-Layer-Perceptron Classifier')

    fig.tight_layout()
    # plt.show()

    plt.savefig(path_save + 'v_vs_t_results_10kit.png')
    print(path_save + 'v_vs_t_results_10kit.png')
    plt.close()


def line_plot_per_layer(enc_task, path_save, data_dict_v_new, data_dict_t_new, data_dict_v_old, data_dict_t_old,
                        config_dict, df_min, df_max):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 9.6))

    # transpose DFs to access values by layer
    data_dict_t_new = data_dict_t_new.T
    data_dict_v_new = data_dict_v_new.T

    data_dict_t_old = data_dict_t_old.T
    data_dict_v_old = data_dict_v_old.T

    layers_v_list = data_dict_t_new.keys()
    layers_t_list = data_dict_t_new.keys()

    x_axis = np.arange(10, 110, 10).tolist()

    colors_list = ["royalblue", "darkorange", "green", "red", "gold", "turquoise", "magenta"]
    dot_symbols_list = ['o', 'v', '*', 'x', '^', 'H', 'D']

    print('data_dict_v_new', data_dict_v_new)
    noise_type = data_dict_v_new['l1'].keys()[0].split('_')[0]

    if config_dict['sent_word_prob'] == 'sent':
        col_para = 'avg'
    elif config_dict['sent_word_prob'] == 'word':
        col_para = enc_task
    else:
        print('Please check your config-file.\n')
        sys.exit()

    for layer_v, color, symbol in zip(layers_v_list, colors_list, dot_symbols_list):
        v_list = [data_noise - data_dict_v_old[layer_v][col_para] for data_noise in data_dict_v_new[layer_v].values]
        ax1.plot(x_axis, [data_noise - data_dict_v_old[layer_v][col_para] for data_noise in
                          data_dict_v_new[layer_v].values], label="Layer " + str(layer_v[-1]), linestyle="-",
                 color=color, marker=symbol)
        ax1.set_xlabel('Noise %')
        ax1.set_ylabel('Degradation due to noise')
        ax1.set_title('Visual Encoder', fontsize=18)

    for layer_t, color, symbol in zip(layers_t_list, colors_list, dot_symbols_list):
        t_list = [data_noise - data_dict_v_old[layer_t][col_para] for data_noise in data_dict_v_new[layer_t].values]
        ax2.plot(x_axis, [data_noise - data_dict_t_old[layer_t][col_para] for data_noise in
                          data_dict_t_new[layer_t].values], label="Layer " + str(layer_t[-1]), linestyle="-",
                 color=color, marker=symbol)
        ax2.set_xlabel('Noise %')
        ax2.set_ylabel('Degradation due to noise')
        ax2.set_title('Text Encoder', fontsize=18)

    if config_dict['classifier'] == 'mlp':
        classifier = 'MLP'
    elif config_dict['classifier'] == 'lr':
        classifier = 'Logistic Regression'
    else:
        'Wrong classifier...'

    fig.suptitle('Results Clean vs. Noisy Data, ' + noise_type + ', ' + enc_task + ', ' + classifier, fontsize=20)
    ax1.legend()
    ax2.legend()

    fig.tight_layout()

    # ax1.set_ylim([-0.5, 0.03])
    # ax2.set_ylim([-0.5, 0.03])

    print(df_max, df_min)
    ax1.set_ylim([df_max, df_min])
    ax2.set_ylim([df_max, df_min])

    if enc_task == 'bigram_shift':
        ax1.set_ylim([-0.2, 0.03])
        ax2.set_ylim([-0.2, 0.03])
    elif enc_task == 'past_present':
        ax1.set_ylim([-0.5, 0.03])
        ax2.set_ylim([-0.5, 0.03])

    # plt.show()
    plt.savefig(path_save + config_dict['config'] + '_' + config_dict['classifier'] + '_' + enc_task + '_' +
                noise_type + 'v_vs_t_results.png')
    print(path_save + config_dict['config'] + '_' + config_dict['classifier'] + '_' + enc_task + '_' + noise_type +
          '_v_vs_t_results.png')
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_stack.png')
    plt.close()


def bleu_line_plot_per_layer(enc_task, path_save, data_dict_v_new, data_dict_t_new, config_dict,
                             bleu_scores, bleu_mttt, noise_type, df_v_old, df_t_old):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(19.2, 9.6))
    # Set the axes title font size
    plt.rc('axes', titlesize=22)  # Set the axes labels font size
    plt.rc('figure', titlesize=25)

    # add zero noise to df
    data_dict_t_new[noise_type + '_0.0'] = df_t_old.T
    data_dict_v_new[noise_type + '_0.0'] = df_v_old.T
    #print('df_v_new', data_dict_v_new)

    # transpose DFs to access values by layer
    data_dict_t_new = data_dict_t_new.T
    data_dict_v_new = data_dict_v_new.T

    # sort new indeces
    data_dict_t_new = data_dict_t_new.sort_index()
    data_dict_v_new = data_dict_v_new.sort_index()

    #print('data_dict_v_new', data_dict_v_new)
    print('bleu', bleu_scores)

    layers_v_list = data_dict_v_new.keys()
    layers_t_list = data_dict_t_new.keys()

    x_axis = np.arange(0, 110, 10).tolist()

    # colors_list = ["royalblue", "darkorange", "green", "red", "gold", "turquoise", "magenta"]
    colors_list = ['#f9d030', '#b8ee30', '#26dfd0', '#f62aa0', '#b175ff', '#1c9b8e', '#ff8976']
    dot_symbols_list = ['o', 'v', '*', 'x', '^', 'H', 'D']

    for layer_v, color, symbol in zip(layers_v_list, colors_list, dot_symbols_list):
        ax1.plot(x_axis, data_dict_v_new[layer_v],
                   label="Layer " + str(layer_v[-1]), linestyle="-", color=color, marker=symbol, markersize=10)
        ax1.set_xlabel('Noise Level', fontsize=18)
        ax1.set_ylabel('Probing Accuracy', fontsize=18)
        ax1.set_title('Probing Performance' + '\n' + 'Visual Encoder')

    for layer_t, color, symbol in zip(layers_t_list, colors_list, dot_symbols_list):
        ax2.plot(x_axis, data_dict_t_new[layer_t],
                   label="Layer " + str(layer_t[-1]), linestyle="-", color=color, marker=symbol, markersize=10)
        ax2.set_xlabel('Noise Level', fontsize=18)
        ax2.set_ylabel('Probing Accuracy', fontsize=18)
        ax2.set_title('Probing Performance' + '\n' + 'Text Encoder')

    for model, symbol, line in zip(['t', 'v'], dot_symbols_list[:2], ['-', ':']):
        label_text = None
        if model == 'v':
            label_text = 'Visual Encoder'
        elif model == 't':
            label_text = 'Text Encoder'
        ax3.plot(x_axis, bleu_scores.filter(regex='^'+model, axis=1), label=label_text, linestyle=line, color='#4C4C4C',
                 marker=symbol, markersize=10)
        ax3.set_xlabel('Noise Level', fontsize=18)
        ax3.set_ylabel('BLEU Accuracy', fontsize=18)
        ax3.set_title('BLEU WMT')

    for model, symbol, line in zip(['t', 'v'], dot_symbols_list[:2], ['-', ':']):
        label_text = None
        if model == 'v':
            label_text = 'Visual Encoder'
        elif model == 't':
            label_text = 'Text Encoder'
        ax4.plot(x_axis, bleu_mttt.filter(regex='^'+model, axis=1), label=label_text, linestyle=line, color='#4C4C4C',
                 marker=symbol, markersize=10)
        ax4.set_xlabel('Noise Level', fontsize=18)
        ax4.set_ylabel('BLEU Accuracy', fontsize=18)
        ax4.set_title('BLEU MTTT')

    fig.suptitle('Results BLEU Score, Noise Type: ' + noise_type + ', Probing Task: ' + enc_task)
    ax1.legend(loc='upper right', fontsize="10")
    ax2.legend(loc='upper right', fontsize="10")
    ax3.legend(loc='upper right', fontsize="10")
    ax4.legend(loc='upper right', fontsize="10")

    fig.tight_layout()

    upper_bound_y = max([max(df_t_old.T), max(df_v_old.T)]) + 0.01
    lower_bound_y = min([min(data_dict_t_new['l1']), min(data_dict_v_new['l1'])]) - 0.05

    ax1.set_ylim([lower_bound_y, upper_bound_y])
    ax2.set_ylim([lower_bound_y, upper_bound_y])
    ax3.set_ylim([-1, 36])
    ax4.set_ylim([-1, 36])

    # plt.show()
    plt.savefig(path_save + config_dict['config'] + '_' + enc_task + '_' + noise_type + '_v_vs_t_results.png')
    print(path_save + config_dict['config'] + '_' + enc_task + '_' + noise_type + '_v_vs_t_results.png')
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_stack.png')
    plt.close()


def bleu_mttt_line_plot_per_layer(path_save, config_dict, bleu_scores, noise_types_list):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.2, 9.6))
    # Set the axes title font size
    plt.rc('axes', titlesize=22)  # Set the axes labels font size
    plt.rc('figure', titlesize=25)

    x_axis = np.arange(0, 110, 10).tolist()
    print(x_axis)

    colors_list = ["royalblue", "darkorange", "green", "red", "gold", "turquoise", "magenta"]
    dot_symbols_list = ['o', 'v', '*', 'x', '^', 'H', 'D']

    print('bleu_scores', bleu_scores)

    for noise_type, ax in zip(sorted(noise_types_list), [ax1, ax2, ax3]):
        bleu_scores_zero = bleu_scores.filter(regex='_0', axis=1)
        bleu_scores_noise = bleu_scores.filter(regex=noise_type, axis=1)
        for model, color, symbol in zip(['t', 'v'], colors_list[:2], dot_symbols_list[:2]):
            # bleu_scores_noise[model + '_' + noise_type][str(0.0)] = bleu_scores_zero.filter(regex='^'+model)
            bleu_scores_noise = bleu_scores_noise.sort_index()
            print('noise', bleu_scores_noise)
            label_text = None
            if model == 'v':
                label_text = 'Visual Encoder'
            elif model == 't':
                label_text = 'Text Encoder'
            print('model', bleu_scores_noise.filter(regex='^' + model, axis=1))
            ax.plot(x_axis, bleu_scores_zero.filter(regex='^' + model, axis=1), label=label_text, linestyle="-",
                     color=color, marker=symbol)
            ax.plot(x_axis, bleu_scores_noise.filter(regex='^' + model, axis=1), label=label_text, linestyle="-",
                     color=color, marker=symbol)
            ax.set_xlabel('Noise Level', fontsize=18)
            ax.set_ylabel('BLEU Accuracy', fontsize=18)
            ax.set_title('BLEU Performance, Noise Type: ' + noise_type)

    fig.suptitle('Results BLEU Score, Dataset: MTTT, Noise')
    ax1.legend()
    ax2.legend()
    ax3.legend()

    fig.tight_layout()

    # upper_bound_y = max([max(df_t_old.T), max(df_v_old.T)]) + 0.01
    # lower_bound_y = min([min(data_dict_t_new['l1']), min(data_dict_v_new['l1'])]) - 0.01

    ax1.set_ylim([0, 36])
    ax2.set_ylim([0, 36])
    ax3.set_ylim([0, 36])

    plt.show()
    # plt.savefig(path_save + config_dict['config'] + '_MTTT_noise_v_vs_t_results.png')
    # print(path_save + config_dict['config'] + '_MTTT_noise_v_vs_t_results.png')
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_stack.png')
    plt.close()


def one_plot_old_new(enc_task, path_save, data_dict_v_new, data_dict_t_new, data_dict_v_old, data_dict_t_old, config_dict):
    fig, axs = plt.subplots(figsize=(12.8, 9.6))

    print('data_dict_v_new.keys()', data_dict_v_new.keys())

    for col_v in sorted(data_dict_v_new.keys(), reverse=True):
        print('col_v', col_v)

        labels = list(data_dict_v_new[col_v].keys())

        x = np.arange(len(labels))  # the label locations
        print('x', x)
        x_v = np.arange(len(labels))  # the label locations
        x_t = np.arange(len(labels) - 1)  # the label locations
        width = 0.35  # the width of the bars

        # new results
        v_results_new = data_dict_v_new[col_v].values
        t_results_new = data_dict_t_new[col_v].values

        # old results
        v_results_old = data_dict_v_old['avg'].values
        t_results_old = data_dict_t_old['avg'].values

        # bar plots need same length
        if config_dict['config'] != 'noise':
            t_results_new = np.append(t_results_new, 0.0)
            t_results_old = np.append(t_results_old, 0.0)

        if config_dict['config'] == 'oise':
            # results difference
            v_results_diff = [element1 - element2 for (element1, element2) in zip(v_results_old, v_results_new)]
            t_results_diff = [element1 - element2 for (element1, element2) in zip(t_results_old, t_results_new)]
            # print('results difference', v_results_diff, t_results_diff)

            axs.plot(x_v, v_results_diff, label="Visual Model", linestyle="-", color="royalblue")
            axs.plot(x_t, t_results_diff, label="Text Model", linestyle="-", color="darkorange")
            axs.set_xlabel('Layers')
            axs.set_ylabel('Difference between Models')
            axs.set_title('Results Clean vs. Noisy Data, ' + col_v + ', ' + enc_task)
            # axs.set_xticks(x, labels)
            axs.legend()

            # axs.fill_between(labels, v_results_new, t_results_new, color="grey", alpha=0.3)

            fig.tight_layout()
            # axs.set_ylim([0.5, 1.0])

        else:
            axs.bar(x - width / 2, v_results_old, width, label='Visual Model Old', color="lightsteelblue")
            axs.bar(x + width / 2, t_results_old, width, label='Text Model Old', color="bisque")

            axs.bar(x - width / 2, v_results_new, width*0.5, label='Visual Model New', color="royalblue",
                    edgecolor='black')
            axs.bar(x + width / 2, t_results_new, width*0.5, label='Text Model New', color="darkorange",
                    edgecolor='black')

            axs.set_xlabel('Layers')
            axs.set_ylabel('Results Linear Classifier')
            axs.set_title(col_v)
            axs.set_xticks(x, labels)
            axs.legend()

            fig.tight_layout()
            axs.set_ylim([0.3, 1.1])
        # axs[plot_x, plot_y].set_figure(figsize=(1280, 960))

    plt.show()
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_' + str(int(size)) + '_stack3.png')
    # plt.savefig(path_save + enc_task + '_v_vs_t_results_stack.png')
    plt.close()
