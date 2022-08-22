import spacy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def eval_distribution(read_path_file):
    nlp = spacy.load("de_core_news_sm")
    data_dict = defaultdict(int)
    data_perc_dict = defaultdict(float)

    # with open('/home/anastasia/PycharmProjects/xprobe/de/subj_number/subjnum_tr_out.csv') as file:
    with open(read_path_file) as file:
        for line in file:
            feat_list = list()

            doc = nlp(line)
            sb_num_list = [(token.dep_, token.morph.get('Number')[0]) for token in doc if token.dep_ in ['sb'] and
                           token.morph.get('Number')]
            empty_morph_list = list(filter(lambda dep_num: dep_num[1] is [], sb_num_list))
            data_dict[tuple(sb_num_list)] += 1

    print(len(data_dict.keys()), '\n', data_dict)

    return data_dict


def plot_results_pie(enc_task, data_dict, path_save):  # , maj_cl_val):
    # wasn't able to extract normalized Layer for text-model; temporary fix, drop normalized layer for v_model
    # data_dict_v.drop(data_dict_v.tail(1).index, inplace=True)
    sorted_data_dict = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))

    labels = list(sorted_data_dict.keys())
    results = list(sorted_data_dict.values())

    fig1, ax1 = plt.subplots()
    ax1.pie(results, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    total = sum(results)
    plt.legend(
        loc='best',
        labels=['%s, %1.1f%%' % (
            l, (float(s) / total) * 100) for l, s in zip(labels, results)],
        prop={'size': 10},
        bbox_to_anchor=(0.81, 0.5),
        # bbox_transform=fig1.transFigure
    )

    plt.tight_layout()
    plt.show()
    # plt.savefig(path_save + 'eval_' + enc_task + '.png')

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars
    #
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width / 2, results, width)  ## , label='ToDo')
    #
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xlabel('Pairs')
    # ax.set_ylabel('Distribution in Percentage')
    # ax.set_title('Dataset evalutation for the task ' + enc_task)
    # ax.set_xticks(x, labels)
    # ax.legend()
    #
    # ax.bar_label(rects1, padding=3)
    #
    # fig.tight_layout()
    # # plt.ylim([0.45, 1.0])
    #
    # plt.show()
    # plt.savefig(path_save + 'v_vs_t_results_' + enc_task + '_' + str(size) + '.png')
    plt.close()


if __name__ == '__main__':
    # eval_dict, eval_perc_dict = eval_distribution('/local/anasbori/xprobe/de/subj_number/subjnum_tr_out.csv')
    eval_dict = eval_distribution('/home/anastasia/PycharmProjects/xprobe/de/subj_number/Verwaltung_tr.txt')
    # plot_results_pie('SUBJ', eval_perc_dict, '/local/anasbori/visrepProb/task_encs/')
    plot_results_pie('SUBJ', eval_dict, '/local/anasbori/visrepProb/task_encs/')

