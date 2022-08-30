import spacy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def eval_distribution(read_path_file):
    nlp = spacy.load("de_core_news_sm")
    data_dict = defaultdict(int)
    data_str_dict = defaultdict(list)
    data_perc_dict = defaultdict(float)

    # with open('/home/anastasia/PycharmProjects/xprobe/de/subj_number/subjnum_tr_out.csv') as file:
    with open(read_path_file) as file:
        for line in file:
            feat_list = list()

            doc = nlp(line)
            sb_num_list = [(token.dep_, token.morph.get('Number')[0]) for token in doc if token.dep_ in
                           ['sb', 'oa', 'oc', 'og'] and token.morph.get('Number')]
            sub_ob_num_list = list(('o', obj[1]) if obj[0] in ['oa', 'oc', 'og'] else obj for obj in sb_num_list)

            # empty_morph_list = list(filter(lambda dep_num: dep_num[1] is [], sb_num_list))
            data_dict[tuple(set(sub_ob_num_list))] += 1
            data_str_dict[tuple(set(sub_ob_num_list))].append(line)

    print('int_dict', len(data_dict.keys()), 'str_dict', len(data_str_dict.keys()))
    # print(len(data_dict.keys()), '\n', data_dict)
    # print(len(data_str_dict.keys()), '\n', data_str_dict)

    return data_dict, data_str_dict


def save_dict_to_file(data_path, task, list_dict):
    for key in list_dict.keys():
        key_name = "".join("".join("".join('_'.join(str(key).split(', ')).split("'")).split("(")).split(")"))
        with open(data_path + task + '_' + key_name + '.txt', 'w', encoding='utf-8') as f:
            for val in list_dict[key]:
                f.write(val + '\n')


def plot_results_pie(enc_task, data_dict, path_save):  # , maj_cl_val):
    # wasn't able to extract normalized Layer for text-model; temporary fix, drop normalized layer for v_model
    # data_dict_v.drop(data_dict_v.tail(1).index, inplace=True)
    sorted_data_dict = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))

    labels = list(sorted_data_dict.keys())
    results = list(sorted_data_dict.values())

    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.subplots()
    ax1.pie(results, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    total = sum(results)
    plt.legend(
        loc='best',
        labels=['%s, %1.1f%%' % (
            l, (float(s) / total) * 100) for l, s in zip(labels, results)],
        prop={'size': 13},
        bbox_to_anchor=(0.81, 0.5),
        # bbox_transform=fig1.transFigure
    )
    plt.title('Data evaluation of task ' + enc_task)
    plt.tight_layout()
    # plt.show()
    plt.savefig(path_save + 'eval_' + enc_task + '.png')

    plt.close()


if __name__ == '__main__':
    # test locally
    eval_dict, eval_str_dict = eval_distribution('/home/anastasia/PycharmProjects/xprobe/de/subj_number/Verwaltung_tr.txt')
    save_dict_to_file('/home/anastasia/PycharmProjects/visrepProb/task_encs/subj_number/', 'SUBJ', eval_str_dict)
    # plot_results_pie('SUBJ', eval_dict, '/home/anastasia/PycharmProjects/visrepProb/task_encs/')

    # SUBJ
    # eval_dict, eval_str_dict = eval_distribution('/local/anasbori/xprobe/de/subj_number/subjnum_out_clean_uniq.csv')
    # save_dict_to_file('/local/anasbori/visrepProb/task_encs/subj_number/', 'SUBJ', eval_str_dict)
    # plot_results_pie('SUBJ', eval_dict, '/local/anasbori/visrepProb/task_encs/')

    # OBJ
    # eval_dict, eval_str_dict = eval_distribution('/local/anasbori/xprobe/de/obj_number/objnum_out_uniq.csv')
    # save_dict_to_file('/local/anasbori/visrepProb/task_encs/obj_number/', 'OBJ', eval_str_dict)
    # plot_results_pie('OBJ', eval_dict, '/local/anasbori/visrepProb/task_encs/')

