import spacy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedSet
import pickle


def spacy_distribtion(data_list):
    nlp = spacy.load("de_core_news_sm")
    data_dict = defaultdict(int)
    data_str_dict = defaultdict(list)

    data_key_list = list()

    for element in data_list:
        feat_list = list()

        doc = nlp(element)

        sb_num_list = [(token.dep_, token.morph.get('Number')[0]) for token in doc if token.dep_ in
                       ['sb', 'oa', 'oc', 'og'] and token.morph.get('Number')]
        sub_ob_num_list = list(('o', obj[1]) if obj[0] in ['oa', 'oc', 'og'] else obj for obj in sb_num_list)

        # empty_morph_list = list(filter(lambda dep_num: dep_num[1] is [], sb_num_list))
        # data_dict[tuple(set(sub_ob_num_list))] += 1
        key_name = "_".join(sum(tuple(SortedSet(sub_ob_num_list)), ()))

        data_dict[key_name] += 1
        data_str_dict[key_name].append(element)

        data_key_list.append((sub_ob_num_list, set(sub_ob_num_list)))
        # print('data_dict', data_dict)

    return data_dict, data_str_dict


def eval_distribution(read_path_file):
    if read_path_file.split('.')[-1] == 'npy':
        data_arr = np.load(read_path_file, allow_pickle=True)
        return spacy_distribtion(data_arr)
    else:
        with open(read_path_file) as file:
            return spacy_distribtion(file)  # , data_key_list


def save_listelements_to_file(data_path, enc_task, list_dict):
    for key in list_dict.keys():
        key_name = "".join("".join("".join('_'.join(str(key).split(', ')).split("'")).split("(")).split(")"))
        print("data_path + enc_task + '_' + key_name + '.txt'", data_path + enc_task + '_' + key_name + '.txt')
        with open(data_path + enc_task + '_' + key_name + '.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(list_dict[key]))
            # for val in list_dict[key]:
            #     f.write(val)


def save_dict_to_file(out,  data_dict):
    with open(out + '.p', 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def read_in_pickled_dict(data_pickled):
    with open(data_pickled + '.p', 'rb') as fp:
        data = pickle.load(fp)
        return data


def plot_results_pie(enc_task, data_dict, path_save, filename):  # , maj_cl_val):
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
    plt.show()
    # plt.savefig(path_save + 'eval_' + enc_task + '_' + filename + '.png')

    plt.close()


def plot_results_o_s_pie(enc_task, data_dict, path_save, filename, traintest):  # , maj_cl_val):
    # wasn't able to extract normalized Layer for text-model; temporary fix, drop normalized layer for v_model
    # data_dict_v.drop(data_dict_v.tail(1).index, inplace=True)
    sorted_data_dict = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))

    for key_val in ["'sb', 'Plur'", "'sb', 'Sing'", "'o', 'Plur'", "'o', 'Sing'"]:
        filtered_dict = dict(filter(lambda dict_elem: key_val in str(dict_elem[0]), sorted_data_dict.items()))
        sorted_dict = dict(sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True))

        labels = list(sorted_dict.keys())
        results = list(sorted_dict.values())

        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.subplots()
        ax1.pie(results,  startangle=90)
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
        plt.title('Data evaluation of task ' + enc_task + ' ' + traintest + ' ' + key_val)
        plt.tight_layout()
        plt.show()
        # plt.savefig(path_save + 'eval_' + enc_task + '_' + '_'.join(key_val.split(', ')) + '_' +
        #             filename.split('.')[0] + '.png')

        plt.close()


def compare_lists(d_key_list1, d_key_list2):
    first_elem = [x[0] == y[0] for x, y in zip(d_key_list1, d_key_list2)]
    second_elem = [x[1] == y[1] for x, y in zip(d_key_list1, d_key_list2)]

    print(first_elem, '\n', second_elem, '\n')

    with open('/home/anastasia/PycharmProjects/xprobe/compare_lists.txt', 'w', encoding='utf-8') as f:
        for tuple1, tuple2 in zip(d_key_list1, d_key_list2):
            f.write(str(tuple1) + '\t' + str(tuple2) + '\n')


if __name__ == '__main__':
    task_list = ['SUBJ', 'OBJ']

    for task in task_list:
        # test locally
        # file_path = '/home/anastasia/PycharmProjects/xprobe/de/' + task.lower() + '_number/'
        file_path = '/home/anastasia/PycharmProjects/visrepProb/task_encs/' + task.lower() + '_number/'
        file_path_out = '/home/anastasia/PycharmProjects/visrepProb/task_encs/' + task.lower() + '_number/'
        # file_name = 'Verwaltung_tr.txt'

        # server
        # file_path = '/local/anasbori/visrepProb/task_encs/' + task.lower() + '_number/'
        for train_test in ('train', 'test'):
            file_name = train_test + '_raw_sentences.npy'

            # eval_dict, eval_str_dict = eval_distribution(file_path_in + file_name)
            eval_dict, eval_str_dict = eval_distribution(file_path + file_name)

            '''CHECK VALIDITY OF KEYS'''
            # eval_dict, eval_str_dict, key_list = eval_distribution('/home/anastasia/PycharmProjects/xprobe/de/subj_number/'
            #                                                        'Verwaltung_tr.txt')
            # eval_dict, eval_str_dict, key_list2 = eval_distribution('/home/anastasia/PycharmProjects/xprobe/de/subj_number/'
            #                                                         'Verwaltung_tr.txt')

            # compare_lists(key_list, key_list2)

            # save_dict_to_file(file_path_out + 'eval_' + task + '_' + train_test + '_' + file_name, eval_dict)
            save_dict_to_file(file_path + 'eval_' + task + '_' + train_test + '_' + file_name, eval_dict)
            save_listelements_to_file(file_path, 'SUBJ', eval_str_dict)
            # pickled_dict = read_in_pickled_dict(file_path_out + 'eval_' + task + '_' + train_test + '_' + file_name)
            pickled_dict = read_in_pickled_dict(file_path + 'eval_' + task + '_' + train_test + '_' + file_name)

            plot_results_pie(task, pickled_dict, '/home/anastasia/PycharmProjects/visrepProb/task_encs/', file_name)
            # plot_results_o_s_pie(task, pickled_dict, '/home/anastasia/PycharmProjects/visrepProb/task_encs/', file_name,
            #                      train_test)

            ''' SUBJ '''
            # eval_dict, eval_str_dict = eval_distribution('/local/anasbori/xprobe/de/subj_number/subjnum_out_clean_uniq.csv')
            # save_dict_to_file('/local/anasbori/visrepProb/task_encs/subj_number/', 'SUBJ', eval_str_dict)
            # plot_results_pie('SUBJ', eval_dict, '/local/anasbori/visrepProb/task_encs/')
            # plot_results_pie('SUBJ', eval_dict, '/home/anastasia/PycharmProjects/visrepProb/task_encs/')

            ''' OBJ '''
            # eval_dict, eval_str_dict = eval_distribution('/local/anasbori/xprobe/de/obj_number/objnum_out_uniq.csv')
            # save_dict_to_file('/local/anasbori/visrepProb/task_encs/obj_number/', 'OBJ', eval_str_dict)
            # plot_results_pie('OBJ', eval_dict, '/local/anasbori/visrepProb/task_encs/')

