from random import randint, shuffle, sample
import numpy as np
import pandas as pd
import os
import yaml
import sys


class Noise:

    def ab_create_cam(self, string):
        if len(string) > 3:
            c = list(string)
            first_letter = c.pop(0)
            last_letter = c.pop(-1)
            shuffle(c)
            return first_letter + ''.join(c) + last_letter
        else:
            return string

    def ab_create_swap(self, string):
        c = list(string)
        rand_num = randint(0, len(c) - 1)

        if rand_num == len(c) - 1:
            i, j = rand_num - 1, rand_num
            c[i], c[j] = c[j], c[i]
        elif rand_num == 0:
            i, j = rand_num, rand_num + 1
            c[i], c[j] = c[j], c[i]
        else:
            i, j = rand_num, rand_num + 1
            c[i], c[j] = c[j], c[i]

        return ''.join(c)

    def create_noise_in_sent(self, sent, noise_perc, noise_type):
        if int(len(sent) * noise_perc) == 0:
            # TODO !!
            return 'test123'
            # return sent
        else:
            sent_list = sent.split(maxsplit=-1)
            num_noise_words = int(len(sent_list) * noise_perc)
            words_positions_in_sent = sample(range(len(sent_list)), num_noise_words)
            # words_for_noise = [sent[pos] for pos in words_positions_in_sent]

        if noise_type == 'swap':
            for pos in words_positions_in_sent:
                sent_list[pos] = self.ab_create_swap(sent_list[pos])
        elif noise_type == 'cam':
            for pos in words_positions_in_sent:
                sent_list[pos] = self.ab_create_cam(sent_list[pos])
        elif noise_type == 'real':
            pass
        elif noise_type == 'random':
            pass
        else:
            return 'This noise type is not support, please choose a different one.'

        return ' '.join(sent_list)


if __name__ == '__main__':
    with open('config_visrep.yml') as config:
        tasks_dict = yaml.load(config, Loader=yaml.FullLoader)
        path_server_o_lokal = None

        if sys.argv[1] == 's':
            path_server_o_lokal = tasks_dict['server_path']
        elif sys.argv[1] == 'l':
            path_server_o_lokal = tasks_dict['lokal_path']

        # task_list = ['TENSE', 'OBJ', 'SUBJ', 'BIGRAM']
        # task_list = ['subj_number', 'obj_number', 'past_present', 'bigram_shift']
        task_list = tasks_dict['tasks']

        for task in task_list:
            # path_in = '/home/anastasia/PycharmProjects/visrepProb/task_encs/subj_number/test_raw_sentences.npy'
            path_file_in = path_server_o_lokal + tasks_dict['data_path_in'] + task + '/' \
                           + tasks_dict['noise_test_file_in']
            noise_type_list = tasks_dict['noise_type']
            noise_perc_list = tasks_dict['noise_perc']
            # path_in = '/home/anastasia/PycharmProjects/xprobe/de/subj_number/Verwaltung_tr_no_Labels.npy'

            path_out = path_server_o_lokal + tasks_dict['data_path_in'] + task + '/'
            file_out = tasks_dict['noise_test_file_out']

            try:
                os.mkdir(path_out)
            except OSError as error:
                # print(error)
                pass

            noise = Noise()

            # load np-array to pd df
            df_all = pd.DataFrame(np.load(path_file_in, allow_pickle=True))
            df_all.columns = ['raw']

            df_noise = pd.DataFrame()

            for noise_type in noise_type_list:
                for noise_perc in noise_perc_list:

                    col_name = noise_type + '_' + str(noise_perc)
                    df_noise[col_name] = df_all.apply(lambda row: noise.create_noise_in_sent(row['raw'], noise_perc,
                                                                                             noise_type), axis=1)

            df_noise.to_csv(path_out + file_out, index=False, encoding='utf-8', sep =';')
            # np.savetxt(path_out + file_out, df_noise, delimiter=';')

