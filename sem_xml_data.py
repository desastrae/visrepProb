import xml.etree.ElementTree as ET
import os
import itertools


def get_sem_data_dirs(folder):
    # folder = '/home/anastasia/PycharmProjects/visrepProb/word_level/pmb-sample-4.0.0/data/de/'
    # folder = path_dir + 'sem/pmb-sample-4.0.0/data/de/'

    print(folder + 'gold/')

    gold_sub_folders = [['gold/' + name + '/' + f_dir for f_dir in os.listdir(folder + 'gold/' + name + '/')]
                        for name in os.listdir(folder + 'gold/')]
    silver_sub_folders = [['silver/' + name + '/' + f_dir for f_dir in os.listdir(folder + 'silver/' + name + '/')]
                          for name in os.listdir(folder + 'silver/')]
    bronze_sub_folders = [['bronze/' + name + '/' + f_dir for f_dir in os.listdir(folder + 'bronze/' + name + '/')]
                          for name in os.listdir(folder + 'bronze/')]

    gold_dir = list(itertools.chain(*gold_sub_folders[0]))
    silver_dir = list(itertools.chain(*silver_sub_folders[0]))
    bronze_dir = list(itertools.chain(*bronze_sub_folders[0]))

    # print(len(gold_sub_folders), len(silver_sub_folders), len(bronze_sub_folders))
    # print(len(gold_sub_folders))
    # print(gold_sub_folders)
    print('gold', len(gold_dir), 'silver', len(silver_dir[0]), 'bronze', len(bronze_dir[0]))

    return gold_dir, silver_dir, bronze_dir

# sem_val = True  # False

# if sem_val:


def get_train_test_sem(folder, gold_dir, silver_dir, bronze_dir):
    # folder = '/home/anastasia/PycharmProjects/visrepProb/word_level/pmb-sample-4.0.0/data/de/'

    train_data_list = list()
    test_data_list = list()

    for data_dir, train_val, test_val in ((gold_dir, (0, 2282, 'train'), (2282, 3043, 'test')),
                                          (silver_dir, (0, 4916, 'train'), (4916, 6554, 'test')),
                                          (bronze_dir, (0, 302, 'train'), (302, 403, 'test'))):

    # for data_dir, train_val, test_val in ((gold_dir, (0, 7, 'train'), (7, 10, 'test')),
    #                                       (gold_dir, (0, 7, 'train'), (7, 10, 'test')),
    #                                       (gold_dir, (0, 7, 'train'), (7, 10, 'test'))):

        for data in (train_val, test_val):
            data_set = list()

            for file_dir in data_dir[data[0]:data[1]]:
                sent_list = list()

                tree = ET.parse(folder + file_dir + '/de.drs.xml')
                root = tree.getroot()

                save_tok = None

                for node in tree.iter('tag'):
                    # test = str(node.attrib).split('}')
                    test = str(node.attrib)
                    # print(test)
                    # if 'sem' in test[0]:

                    if 'tok' in test:
                        # print(test[1])
                        # print(node.attrib, node.text)
                        save_tok = node.text
                    if 'sem' in test:
                        # print(test[1])
                        # print(node.attrib, node.text)
                        if save_tok != 'ø':
                            sent_list.append((save_tok, node.text))

                data_set.append(sent_list)

            if data[2] == 'train':
                train_data_list.extend(data_set)
            elif data[2] == 'test':
                test_data_list.extend(data_set)

    # print(train_data_list, '\n', test_data_list)
    print('train: ', len(train_data_list))
    print('test: ', len(test_data_list))

    return train_data_list, test_data_list


# gold, silver, bronze = get_sem_data_dirs('test')
# train, test = get_train_test_set_sem(gold, silver, bronze)
