import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import yaml
from natsort import natsorted
from os import walk, listdir
from scipy import stats
import seaborn as sns

print(mpl.style.available)


def apply_relative_change_task(clean_file, noise_file, word_sent):
    clean_df = pd.read_csv(clean_file, index_col=0)
    noise_df = pd.read_csv(noise_file, index_col=0)

    if word_sent == 'word':
        w_s_clean = clean_df['dep']
    elif word_sent == 'sent':
        w_s_clean = clean_df['past_present']
    # print(clean_df.loc['l1'])
    # print(noise_df.loc['l1'])
    layers = list(clean_df.index)

    for l in layers:
        # print("noise_df['cmabrigde_0.1'].loc[l]", noise_df['cmabrigde_0.1'].loc[l])
        # print("w_s_clean.loc[l]", w_s_clean.loc[l])
        noise_df.loc[l] = noise_df.loc[l].apply(lambda n: (n - w_s_clean.loc[l]) / w_s_clean.loc[l])
        # print("noise_df['cmabrigde_0.1'].loc[l]", noise_df['cmabrigde_0.1'].loc[l])

    return noise_df


def relative_change_bleu(data, m_list):
    bleu_df = pd.read_csv(data, index_col=0)
    # print(bleu_df.keys())
    for m in m_list:
        clean_bleu_score = bleu_df[m + '0.0'].loc['bleu']
        m_noise_list = list(filter(lambda n: m in n, bleu_df.keys()))[1:]
        # print(m_noise_list)
        # print(clean_bleu_score)
        # print(bleu_df)
        for noise in m_noise_list:
            # print(bleu_df[noise], clean_bleu_score)
            bleu_df[noise] = bleu_df[noise].map(lambda n: (n - clean_bleu_score) / clean_bleu_score)
            # print(bleu_df[noise])
    bleu_df = bleu_df.drop(['t_0.0', 'v_0.0'], axis=1)
    bleu_df = bleu_df.drop(['bleu'], axis=0)
    return bleu_df


def plot_correlation_scores(t_prob_df, v_prob_df, bleu_WMT_df, bleu_MTTT_df, noise_type, w_s, out_path, task, classifier):
    if classifier == 'lr':
        cla_model = 'Linear Regression'
    elif classifier == 'mlp':
        cla_model = 'Multi-Layer Perceptron'
    # 'seaborn-pastel'
    # fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(12.8, 9.6))
    colors = sns.color_palette('colorblind')
    sequential_colors = sns.color_palette("Blues", 15)
    # sequential_colors = sns.color_palette("ch:start=-.3,rot=.2", 10)
    # sequential_colors2 = sns.color_palette("RdPu_r", 20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 9.6))
    ax1.set_prop_cycle('color', colors)
    ax2.set_prop_cycle('color', colors)

    # print('t: ', t_prob_df.keys(), t_prob_df.index)
    # print('v: ', v_prob_df.keys(), v_prob_df.index)

    # Set the axes title font size
    plt.rc('axes', titlesize=22)  # Set the axes labels font size
    # Set the font size of the figure title
    plt.rc('figure', titlesize=25)

    fig.suptitle('Noise / BLEU correlation results, Noise Type: ' + noise_type + ', ' + w_s + '-Level\n'
                 + 'Task: ' + task + ', Classifier: ' + cla_model)

    dot_symbols_list = ['o', 'v', '*', 'x', '^', 'H', 'D']

    ax1.set_xlabel(r'$\Delta$ BLEU', fontsize=18)
    ax2.set_xlabel(r'$\Delta$ BLEU', fontsize=18)
    ax1.set_ylabel(r'$\Delta$ ACC', fontsize=18)
    ax2.set_ylabel(r'$\Delta$ ACC', fontsize=18)

    ax1.set_title('WMT dataset')
    ax2.set_title('MTTT dataset')

    layers_list = v_prob_df.index

    # print(t_prob_df.filter(regex='l33t'))

    # for noise, color, symbol in noise_list, colors_list, dot_symbols_list:
    # for noise in noise_list:
    t_noise_df = t_prob_df.filter(regex=noise_type)
    v_noise_df = v_prob_df.filter(regex=noise_type)

    noise_level_list = [t.split('_')[1] for t in t_noise_df.keys()]
    print('noise_level_list: ', noise_level_list)

    # print(t_noise_df)
    # t_corr = [stats.spearmanr(bleu_df['t_' + noise_type], t_noise_df.loc[layer])[0] for layer in list(t_prob_df.index)]
    # v_corr = [stats.spearmanr(bleu_df['v_' + noise_type], v_noise_df.loc[layer])[0] for layer in list(v_prob_df.index)]
    # print(t_corr)

    t_corr = list()
    # print(noise_df)
    add_labels = True

    for layer in list(t_prob_df.index):
        # print(list(bleu_df['t_l33t']), list(t_noise_df.loc[layer]))
        coef_WMT, p_WMT = stats.spearmanr(bleu_WMT_df['v_' + noise_type], t_noise_df.loc[layer])
        coef_MTTT, p_MTTT = stats.spearmanr(bleu_MTTT_df['v_' + noise_type], t_noise_df.loc[layer])
        # print(layer, t_noise_df.loc[layer])
        # print(coef, p)
        # t_corr.append(res.statistic)
        # print(bleu_df['t_' + noise_type])

        # ax1.scatter(list(bleu_df['t_' + noise_type]), list(t_noise_df.loc[layer]), color=colors[0])
        # ax1.scatter(list(bleu_df['v_' + noise_type]), list(v_noise_df.loc[layer]), color=colors[5])

        for i in range(len(v_noise_df.loc[layer])):
            # plotting the corresponding x with y
            # and respective color
            # if layer == 'l5' and i == '5':
            if add_labels and i == 5:
                ax1.scatter(bleu_WMT_df['t_' + noise_type][i], t_noise_df.loc[layer][i], color=sequential_colors[i+5],
                            label='Text Model')
                ax1.scatter(bleu_WMT_df['v_' + noise_type][i], v_noise_df.loc[layer][i], color=sequential_colors[i+5],
                            label='Visual Model', marker='x')
                ax2.scatter(bleu_MTTT_df['t_' + noise_type][i], t_noise_df.loc[layer][i], color=sequential_colors[i+5],
                            label='Text Model')
                ax2.scatter(bleu_MTTT_df['v_' + noise_type][i], v_noise_df.loc[layer][i], color=sequential_colors[i+5],
                            label='Visual Model', marker='x')
                # plt.colorbar(cm.ScalarMappable(cmap=cm.blues))
                add_labels = False
            else:
                ax1.scatter(bleu_WMT_df['t_' + noise_type][i], t_noise_df.loc[layer][i], color=sequential_colors[i+5])
                ax1.scatter(bleu_WMT_df['v_' + noise_type][i], v_noise_df.loc[layer][i], color=sequential_colors[i+5], marker='x')
                ax2.scatter(bleu_MTTT_df['t_' + noise_type][i], t_noise_df.loc[layer][i], color=sequential_colors[i+5])
                ax2.scatter(bleu_MTTT_df['v_' + noise_type][i], v_noise_df.loc[layer][i], color=sequential_colors[i+5], marker='x')
            # plt.scatter(x[i], y[i], c=col[i], s=10, linewidth=0)
            ax1.scatter(bleu_WMT_df['v_' + noise_type][i], v_noise_df.loc['l7'][i], color=sequential_colors[i+5], marker='x')
            ax2.scatter(bleu_MTTT_df['v_' + noise_type][i], v_noise_df.loc['l7'][i], color=sequential_colors[i+5], marker='x')

        # if layer == 'l1':
        #     ax1.scatter(list(bleu_df['t_' + noise_type]), list(t_noise_df.loc[layer]), color=colors[0], label='Text Model')
        #     ax1.scatter(list(bleu_df['v_' + noise_type]), list(v_noise_df.loc[layer]), color=colors[5], label='Visual Model')
        # else:
        #     ax1.scatter(list(bleu_df['t_' + noise_type]), list(t_noise_df.loc[layer]), color=colors[0])
        #     ax1.scatter(list(bleu_df['v_' + noise_type]), list(v_noise_df.loc[layer]), color=colors[5])

        t_noise_df.loc['avg'] = t_noise_df.mean(axis=0)
    v_noise_df.loc['avg'] = v_noise_df.mean(axis=0)

    # print(t_noise_df.loc['avg'])
    # print(bleu_df['t_' + noise_type])

    WMT_t_slope, WMT_t_intercept, r_value, p_value, std_err = stats.linregress(bleu_WMT_df['t_' + noise_type], t_noise_df.loc['avg'])
    WMT_v_slope, WMT_v_intercept, r_value, p_value, std_err = stats.linregress(bleu_WMT_df['v_' + noise_type], v_noise_df.loc['avg'])
    MTTT_t_slope, MTTT_t_intercept, r_value, p_value, std_err = stats.linregress(bleu_MTTT_df['t_' + noise_type], t_noise_df.loc['avg'])
    MTTT_v_slope, MTTT_v_intercept, r_value, p_value, std_err = stats.linregress(bleu_MTTT_df['v_' + noise_type], v_noise_df.loc['avg'])

    # Plot regression line
    ax1.plot(bleu_WMT_df['t_' + noise_type], WMT_t_slope*bleu_WMT_df['t_' + noise_type] + WMT_t_intercept, color=colors[9])
    ax1.plot(bleu_WMT_df['v_' + noise_type], WMT_v_slope*bleu_WMT_df['v_' + noise_type] + WMT_v_intercept, color=colors[7])
    ax2.plot(bleu_MTTT_df['t_' + noise_type], MTTT_t_slope*bleu_MTTT_df['t_' + noise_type] + MTTT_t_intercept, color=colors[9])
    ax2.plot(bleu_MTTT_df['v_' + noise_type], MTTT_v_slope*bleu_MTTT_df['v_' + noise_type] + MTTT_v_intercept, color=colors[7])

    # ax1.colorbar(sequential_colors, ax=noise_level_list)

    ax1.legend()
    ax2.legend()

    # ax1.secondary_xaxis('top')

    # ax1.scatter([bleu_df['t_l33t'], t_prob_df.loc[])

    # for noise, color, symbol in noise_list, colors_list, dot_symbols_list:
        # ax1.scatter([bleu_df['t_' + noise], t_prob_df[noise + '_' + noise_perc]) #, linestyle='-', color=color, marker=symbol)
        # ax2.plot(list(bleu_df.index), file_t[task], linestyle='-', color=color, marker=symbol)

    fig.tight_layout()
    # plt.show()
    plt.savefig(out_path + 'pngs/' + classifier + '_' + task + '_' + w_s + '_' + noise_type + '_corr_results.png')
    # plt.savefig(out_path + 'pdfs/' + classifier + '_' + task + '_' + w_s + '_' + noise_type + '_corr_results.pdf')
    plt.close()


if __name__ == '__main__':
    with open('config_visrep.yml') as config:
        config_dict = yaml.load(config, Loader=yaml.FullLoader)
        bleu_corr_path = '/home/anastasia/PycharmProjects/visrepProb/bleu_correlation/'
        files_list = natsorted(next(walk(bleu_corr_path), (None, None, []))[2])

        noise_types_list = config_dict['noise_type']
        m_type_list = ['v_', 't_']
        # bleu_files_list = list(filter(lambda bleu: 'bleu' in bleu, files_list))
        WMT_bleu_file = list(filter(lambda bleu: 'WMT' in bleu, files_list))[0]
        MTTT_bleu_file = list(filter(lambda bleu: 'MTTT' in bleu, files_list))[0]

        bleu_rel_change_WMT_df = relative_change_bleu(bleu_corr_path + WMT_bleu_file, m_type_list)
        bleu_rel_change_MTTT_df = relative_change_bleu(bleu_corr_path + MTTT_bleu_file, m_type_list)

        for w_s in ['word', 'sent'][:1]:
            for clas in ['lr', 'mlp']:
                if w_s == 'word':
                    task_list = config_dict['tasks_word']
                    bleu_corr_path_w_s = '/home/anastasia/PycharmProjects/visrepProb/bleu_correlation/word_csv/'
                    files_list_w_s = natsorted(next(walk(bleu_corr_path_w_s), (None, None, []))[2])
                    t_mtype_files = list(filter(lambda c: clas in c, list(filter(lambda t: 't_' in t, files_list_w_s))))
                    v_mtype_files = list(filter(lambda c: clas in c, list(filter(lambda t: 'v_' in t, files_list_w_s))))
                elif w_s == 'sent':
                    task_list = config_dict['tasks']
                    bleu_corr_path_w_s = '/home/anastasia/PycharmProjects/visrepProb/bleu_correlation/sent_csv/'
                    files_list_w_s = natsorted(next(walk(bleu_corr_path_w_s), (None, None, []))[2])
                    t_mtype_files = list(filter(lambda t: 't_' in t, files_list_w_s))
                    v_mtype_files = list(filter(lambda t: 'v_' in t, files_list_w_s))
                elif w_s == 'sent' and clas == 'mlp':
                    continue
                print(w_s, task_list)
                files_list_w_s = natsorted(next(walk(bleu_corr_path_w_s), (None, None, []))[2])
                t_mtype_files = list(filter(lambda t: 't_' in t, files_list_w_s))
                v_mtype_files = list(filter(lambda t: 'v_' in t, files_list_w_s))
                print(t_mtype_files)

                for task in task_list:
                    t_clean_file = list(filter(lambda clean: 'noise' not in clean, t_mtype_files))[0]
                    print(task)
                    t_noise_file = list(filter(lambda w: task in w, filter(lambda noise: 'noise' in noise,
                                                                           t_mtype_files)))[0]
                    t_rel_change_df = apply_relative_change_task(bleu_corr_path_w_s + t_clean_file,
                                                                 bleu_corr_path_w_s + t_noise_file, w_s)
                    v_clean_file = list(filter(lambda clean: 'noise' not in clean, v_mtype_files))[0]
                    v_noise_file = list(filter(lambda w: task in w, filter(lambda noise: 'noise' in noise,
                                                                           v_mtype_files)))[0]
                    v_rel_change_df = apply_relative_change_task(bleu_corr_path_w_s + v_clean_file,
                                                                 bleu_corr_path_w_s + v_noise_file, w_s)

                    for noise in noise_types_list:
                        if w_s == 'word':
                            plot_correlation_scores(t_rel_change_df, v_rel_change_df, bleu_rel_change_WMT_df,
                                                    bleu_rel_change_MTTT_df, noise, 'Word', bleu_corr_path, task,
                                                    clas)
                            # print(bleu_rel_change_df)
                        elif w_s == 'sent':
                            plot_correlation_scores(t_rel_change_df, v_rel_change_df, bleu_rel_change_WMT_df,
                                                    bleu_rel_change_MTTT_df, noise, 'Sentence', bleu_corr_path,
                                                    task, 'lr')
