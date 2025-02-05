# Load the model in python
from fairseq.models.visual import VisualTextTransformerModel
from fairseq.models.transformer import TransformerModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import os
from os import walk, listdir
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
import pickle
import yaml
from get_image_slices_clean import get_wordpixels_in_pic_slice, get_pic_num_for_word
from get_bpe_word import ref_bpe_word
from conllu import parse
from collections import defaultdict
# import sys
# np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


class VisRepEncodings:
    def __init__(self, config_dict, file_path, path_save_encs, task):
        self.config_dict = config_dict
        self.model = None
        self.encoded_data = None
        self.file_path = file_path
        self.task = task
        if self.config_dict['config'] == 'noise':
            self.data = file_path + self.config_dict['noise_test_file_out']
        else:
            self.data = file_path + self.config_dict['test_file_out']
        self.train_dict = defaultdict
        self.m_para = None
        self.path_save = path_save_encs
        self.path_save_encs = None
        self.create_layer_path = True
        self.all_layers = None

    def make_vis_model(self, para):
        self.model = VisualTextTransformerModel.from_pretrained(
            checkpoint_file='tr_models/visual/WMT_de-en/checkpoint_best.pt',
            target_dict='tr_models/visual/WMT_de-en/dict.en.txt',
            target_spm='tr_models/visual/WMT_de-en/spm.model',
            src='de',
            image_font_path='fairseq/data/visual/fonts/NotoSans-Regular.ttf'
        )
        self.model.eval()  # disable dropout (or leave in train mode to finetune)
        self.m_para = para

        self.path_save_encs = self.path_save + self.m_para + '/'

        try:
            os.mkdir(self.path_save_encs)
        except OSError as error:
            # print(error)
            pass

    def make_text_model(self, para):
        self.model = TransformerModel.from_pretrained(
            'tr_models/text/WMT_de-en_text/',
            checkpoint_file='checkpoint_best.pt',
            src='de',
            tgt='en',
            bpe='sentencepiece',
            # sentencepiece_model='tr_models/de-en/spm_de.model',
            sentencepiece_model='tr_models/text/WMT_de-en_text/spm.model',
            target_dict='dict.en.txt'
        )
        self.model.eval()  # disable dropout (or leave in train mode to finetune)
        self.m_para = para

        self.path_save_encs = self.path_save + self.m_para + '/'

        try:
            os.mkdir(self.path_save_encs)
        except OSError as error:
            # print(error)
            pass

    def save_raw_data(self, df_raw_combined, test_train):
        # print(str(self.path_save) + test_train + '_raw_sentences.npy')
        with open(str(self.path_save) + test_train + '_raw_sentences.npy', 'wb') as f:
            try:
                np.save(f, np.array(df_raw_combined[0]), allow_pickle=True)
            except FileExistsError as error:
                print(error)
                # pass

        with open(str(self.path_save) + test_train + '_raw_labels.npy', 'wb') as f:
            try:
                np.save(f, np.array(df_raw_combined[1]), allow_pickle=True)
            except FileExistsError as error:
                print(error)
                # pass

    def read_in_raw_data(self, d_size, train_or_test_size, test_train, save_df):
        print('\nRead in raw data...\n')

        half_data_size = int(d_size*train_or_test_size / 2)
        df_all = pd.read_csv(self.data, delimiter='\t', header=None)
        df_values = set(df_all[1].values)

        # print(df_values)

        df_combined = pd.DataFrame(columns=df_all.columns.values)

        for val in df_values:
            df_val = df_all[df_all[1] == val].sample(half_data_size)
            df_combined = pd.concat([df_combined, df_val], axis=0)

        df_combined.reset_index(inplace=True, drop=True)

        # print('df_combined', df_combined[1].value_counts(), len(df_combined))

        if save_df:
            self.save_raw_data(df_combined, test_train)

        return df_combined

    def read_pos_raw_data(self, path_file_name):
        pos_tags_file_list = list()
        sentence_list = list()

        # with open('de-utb/de-train.tt') as file:
        with open(path_file_name, 'r', encoding='utf-8') as file:
            for line in file:
                if line == '\n':
                    pos_tags_file_list.append(sentence_list)
                    sentence_list = list()
                    if len(pos_tags_file_list) == self.config_dict['dataset_size'][0]:
                        break
                    continue
                sentence_list.append(tuple(line.split()))
        return pos_tags_file_list

    def read_UD_data(self, path_file):
        dep_data_list = list()

        # with open('word_level/UD_dataset/test30.conllu', "r", encoding="utf-8") as data_file:
        with open(path_file, "r", encoding="utf-8") as data_file:
            data = data_file.read()
            sentences = parse(data)
            for sentence in sentences:
                sent_token_list = list()
                # print(sentence.metadata['text'])
                # print(sentence.default_fields)
                for token in sentence:
                    # print(token['id'], token, token['head'], token['deprel'])
                    if token['xpos'] is None:
                        sent_token_list.append(
                            (token['id'], token['form'], token['head'], token['deprel'], token['upos'],
                             '_'))
                    else:
                        sent_token_list.append((token['id'], token['form'], token['head'], token['deprel'], token['upos'],
                                            token['xpos']))
                # dep_data_dict[sentence.metadata['sent_id']] = sent_token_list
                dep_data_list.append(sent_token_list)

                if len(dep_data_list) == self.config_dict['dataset_size'][0]:
                    break
        # print(dep_data_list)
        return dep_data_list

    def save_label_data(self, data_array, task, tr_or_te):
        # remove init value...
        data_array = np.delete(data_array, 0)
        print('len(data_array): ', len(data_array))

        with open(self.path_save_encs + tr_or_te + '_' + task + '_all_labels_array.npy', 'wb') as f:
            try:
                # print('labels, np.where: ', len(np.where(np.isnan(collected_np_label_array))))
                np.save(f, data_array, allow_pickle=True)
            except FileExistsError as error:
                # print(error)
                pass

    # Translate sentences
    def translate_save(self, batch, tr_or_te, data_name):
        print('\n\nTranslating sentences // Creating encodings...\n\n')

        make_directories = True

        for idx, sent in tqdm(enumerate(batch)):
            translation, layer_dict = self.model.translate(sent)
            # print('translation', translation)

            if make_directories:
                make_directories = False
                self.make_directories(layer_dict, tr_or_te, data_name)

            self.save_encodings(layer_dict, idx, tr_or_te, data_name)

    # Translate sentences for noise
    def translate_save_noise(self, batch, tr_or_te, file_name):
        print('\n\nTranslating sentences // Creating encodings...\n\n')
        self.make_noise_directories(file_name)

        for idx, sent in tqdm(enumerate(batch)):
            # print('sent: ', sent)
            translation, layer_dict = self.model.translate(sent)
            # print('translation', translation)

            if self.m_para == 'v':
                pic_num_words = get_pic_num_for_word(get_wordpixels_in_pic_slice(sent))
                # print('pic_num_words: ', pic_num_words)
                self.save_word_level_encodings(layer_dict, idx, tr_or_te, self.task, pic_num_words, file_name) #, zipped_data_list[2])
            # print('translation', translation)
            elif self.m_para == 't':
                sent_bpe_list = ref_bpe_word(list(sent.split()))
                # print('sent_bpe_list: ', sent_bpe_list)
                # count_sent_bpe_list += len(sent_bpe_list)
                self.save_word_level_encodings(layer_dict, idx, tr_or_te, self.task, sent_bpe_list, file_name) #, zipped_data_list[2])

    def translate_word_level_save(self, batch, tr_or_te, data_name):
        print('\n\nTranslating sentences // Creating encodings...\n\n')

        make_directories = True

        if data_name == 'dep':
            id_labels_array = np.array(('NONE'))
            head_labels_array = np.array(('NONE'))
            dep_labels_array = np.array(('NONE'))
            upos_labels_array = np.array(('NONE'))
            xpos_labels_array = np.array(('NONE'))
        elif data_name == 'sem':
            sem_labels_array = np.array(('NONE'))

        for idx, sent_data in tqdm(enumerate(batch)):
            # print(sent_data)
            # sent_list, pos_list = list(zip(*sent))
            data_tuple_list = list(zip(*sent_data))

            if data_name == 'dep':
                id_labels_array = np.append(id_labels_array, data_tuple_list[0])
                head_labels_array = np.append(head_labels_array, data_tuple_list[2])
                dep_labels_array = np.append(dep_labels_array, data_tuple_list[3])
                upos_labels_array = np.append(upos_labels_array, data_tuple_list[4])
                xpos_labels_array = np.append(xpos_labels_array, data_tuple_list[5])
            elif data_name == 'sem':
                sem_labels_array = np.append(sem_labels_array, data_tuple_list[1])

            sent = ' '.join(data_tuple_list[0])
            translation, layer_dict = self.model.translate(sent)

            # print('len(layer_dict[l1])', layer_dict['l1'].shape)
            # for key, item in layer_dict.items():
            #     print('np.where: ', np.where(np.isnan(layer_dict[key])))
            if make_directories:
                print('Make directories...')
                make_directories = False
                self.make_directories(layer_dict)

            if self.m_para == 'v':
                pic_num_words = get_pic_num_for_word(get_wordpixels_in_pic_slice(sent))
                self.save_word_level_encodings(layer_dict, idx, tr_or_te, data_name, pic_num_words) #, zipped_data_list[2])
            # print('translation', translation)
            elif self.m_para == 't':
                sent_bpe_list = ref_bpe_word(list(data_tuple_list[0]))
                # count_sent_bpe_list += len(sent_bpe_list)
                self.save_word_level_encodings(layer_dict, idx, tr_or_te, data_name, sent_bpe_list) #, zipped_data_list[2])

            # if idx % 50 == 0:
            #     print('\ndep: ', count_dep, '\nupos: ', count_upos, '\nxpos: ', count_xpos, '\nwords: ', count_words,
            #           '\nsent_bpe_list: ', count_sent_bpe_list)

        if data_name == 'dep':
            self.save_label_data(id_labels_array, 'id', tr_or_te)
            self.save_label_data(head_labels_array, 'head', tr_or_te)
            self.save_label_data(dep_labels_array, 'dep', tr_or_te)
            self.save_label_data(xpos_labels_array, 'xpos', tr_or_te)
            self.save_label_data(upos_labels_array, 'upos', tr_or_te)
        elif data_name == 'sem':
            self.save_label_data(sem_labels_array, 'sem', tr_or_te)

    def translate_test_save(self, batch, tr_or_te, data_name):
        print('\n\nTranslating sentences // Creating encodings...\n\n')

        for idx, sent_data in tqdm(enumerate(batch)):
            data_tuple_list = list(zip(*sent_data))
            print('data_tuple_list', data_tuple_list)

            sent = ' '.join(data_tuple_list[0])
            translation, layer_dict = self.model.translate(sent)
            print('translation', translation)

            if self.m_para == 'v':
                pic_num_words = get_pic_num_for_word(get_wordpixels_in_pic_slice(sent))
                # self.save_word_level_encodings(layer_dict, idx, tr_or_te, data_name, pic_num_words) #, zipped_data_list[2])
            elif self.m_para == 't':
                sent_bpe_list = ref_bpe_word(list(data_tuple_list[1]))
                # count_sent_bpe_list += len(sent_bpe_list)
                # self.save_word_level_encodings(layer_dict, idx, tr_or_te, data_name, sent_bpe_list) #, zipped_data_list[2])

    def translate_process(self, batch):
        collect_layer_dicts = defaultdict(list)
        collect_idx_sent_dict = defaultdict(list)

        for idx, sent in tqdm(enumerate(batch[0])):
            translation, layer_dict = self.model.translate(sent)
            for key_layer, val_enc in layer_dict.items():
                collect_layer_dicts[key_layer].append(np.mean(val_enc.numpy(), axis=0))
                collect_idx_sent_dict[key_layer].append(sent)

        return collect_layer_dicts, collect_idx_sent_dict

    # Create directories for layers
    def make_directories(self, np_dict):
        # print('self.path_save_encs', self.path_save_encs)
        self.all_layers = np_dict.keys()
        print('self.all_layers', self.all_layers)
        print('np_dict.keys()', np_dict.keys())
        data_name = ['train', 'test']

        for tr_or_te in data_name:
            try:
                os.mkdir(self.path_save_encs)
            except OSError as error:
                # print(error)
                pass

            try:
                os.mkdir(self.path_save_encs + tr_or_te + '/')
            except OSError as error:
                # print(error)
                pass

            try:
                os.mkdir(self.path_save_encs + tr_or_te + '/layers/')
                os.mkdir(self.path_save_encs + tr_or_te + '/results/')
            except OSError as error:
                # print(error)
                pass

        # try:
        #     os.mkdir(self.path_save_encs + tr_or_te + '/layers/' + data_name + '/')
        #     os.mkdir(self.path_save_encs + tr_or_te + '/results/' + data_name + '/')
        # except OSError as error:
        #     # print(error)
        #     pass

        print('self.create_layer_path: ', self.create_layer_path)

        if self.create_layer_path:
            for key in np_dict.keys():
                try:
                    os.mkdir(self.path_save_encs + 'train/layers/' + key + '/')
                    print("self.path_save_encs + 'train/layers/' + key + '/'", self.path_save_encs + 'train/layers/' + key + '/')
                except OSError as error:
                    # print(error)
                    pass

        for cl_noi in ['clean/', 'noise/']:
            try:
                os.mkdir(self.path_save_encs + 'test/layers/' + cl_noi)
                os.mkdir(self.path_save_encs + 'test/results/' + cl_noi)
            except OSError as error:
                # print(error)
                pass

            if self.create_layer_path:
                for key in np_dict.keys():
                    try:
                        os.mkdir(self.path_save_encs + 'test/layers/' + cl_noi + key + '/')
                        os.mkdir(self.path_save_encs + 'test/results/' + cl_noi + key + '/')
                    except OSError as error:
                        # print(error)
                        continue

    def make_noise_directories(self, noise_dir):
        # print(self.path_save + self.task + self.m_para + '/test/layers/noise/')
        print('Create noise dirs...')

        layers_folders = natsorted(next(walk(self.path_save + self.m_para + '/test/layers/noise/'),
                                        (None, None, []))[1])
        for layer in layers_folders:
            try:
                os.mkdir(self.path_save + self.m_para + '/' + 'test/layers/noise/' + layer + '/' + noise_dir + '/')
                os.mkdir(self.path_save + self.m_para + '/' + 'test/results/noise/' + noise_dir + '/')
            except OSError as error:
                # print(error)
                continue

    def save_encodings(self, np_dict, sent_num, tr_or_te, d_name):
        # data_name = self.data.split('/')[1].split('.')[0]
        # print(data_name)
        # print('Saving encodings...')
        for key, val in np_dict.items():
            with open(self.path_save_encs + tr_or_te + '/layers/' + d_name + '/' + key + '/' + d_name + '_'
                      + self.m_para + '_sent_' + str(sent_num) + '.npy', 'wb') as f:
                try:
                    np.save(f, val.numpy(), allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

    def save_word_level_encodings(self, np_dict, sent_num, tr_or_te, d_name, pic_num_words_list, noise_type=None):  #, pos_list):
        # data_name = self.data.split('/')[1].split('.')[0]
        # print(data_name)
        # print('Saving encodings...')
        # print('pos_list: ', pos_list, 'pic_num_words_list: ', pic_num_words_list)

        for key, val in np_dict.items():
            np.array(val)
            # print('val: \n', val[0:3])
            # print(np.mean(key_val[1][0:3].numpy(), axis=0))
            # np.save(f, np.mean(val[pic_nums[0]:pic_nums[-1]]), allow_pickle=True)
            count_val = -1
            for word_pic_nums in pic_num_words_list:
                # print('pic_nums', pic_nums)
                count_val += 1
                if tr_or_te == 'test' and self.config_dict['config'] == 'noise':
                    path_tr_te = 'noise/' + key + '/' + noise_type + '/'
                elif tr_or_te == 'test' and self.config_dict['config'] != 'noise':
                    path_tr_te = 'clean/' + key + '/'
                elif tr_or_te == 'train' and self.config_dict['config'] != 'noise':
                    path_tr_te = key + '/'
                else:
                    path_tr_te = ''
                if word_pic_nums[0] == '/':
                    word_file = word_pic_nums[0].replace('/', '"backslash"')
                elif '/' in word_pic_nums[0]:
                    word_file = '"backslash"'
                elif word_pic_nums[0] == '$':
                    word_file = '"dollar"'
                elif word_pic_nums[0] == '€':
                    word_file = '"euro"'
                elif not word_pic_nums[0].isalnum():
                    word_file = '"SUBST"'
                else:
                    word_file = word_pic_nums[0]
                with open(self.path_save_encs + tr_or_te + '/layers/' + path_tr_te + d_name
                          + '_' + self.m_para + '_sent' + str(sent_num) + '_word' + str(count_val) + '_' +
                          word_file + '.npy', 'wb') as f:
                    # print('word_pic_nums[1]: ', word_pic_nums[1], 'len(val): ', len(val))
                    try:
                        # print('Normal')
                        # print(np.mean(val[word_pic_nums[1][0]:word_pic_nums[1][-1]]))
                        # print(len(word_pic_nums[1]))
                        if len(word_pic_nums[1]) == 1:
                            # print('len(word_pic_nums) == 1', word_pic_nums[1][0])
                            # print('save word-level', word_pic_nums[0], 'len(w_pic_nums): ', len(word_pic_nums[1]),
                            #      np.where(np.isnan(val[word_pic_nums[1][0]].numpy())))
                            try:
                                np.save(f, val[word_pic_nums[1]].numpy(), axis=0, allow_pickle=True)
                            except IndexError:
                                np.save(f, val[word_pic_nums[1] - 1].numpy(), axis=0, allow_pickle=True)
                        else:
                            # print('save word-level', word_pic_nums[0], 'len(w_pic_nums): ', len(word_pic_nums[1]),
                            #     np.where(np.isnan(np.mean(val[word_pic_nums[1][0]:word_pic_nums[1][-1]].numpy(),
                            #     axis=0))))
                            np.save(f, np.mean(val[word_pic_nums[1][0]:word_pic_nums[1][-1]].numpy(), axis=0),
                                    allow_pickle=True)
                    # except FileExistsError:
                    #     # print('FileExists Error')
                    #     pass
                    except TypeError:
                        # print('val: ', len(val) - 1, 'word_pic_nums[1]: ', word_pic_nums[1])
                        if isinstance(word_pic_nums[1], int):
                            if word_pic_nums[1] <= (len(val)-1):
                                # print('save word-level', word_pic_nums[0],
                                # np.where(np.isnan(val[word_pic_nums[1]].numpy())))
                                np.save(f, val[word_pic_nums[1]].numpy(), allow_pickle=True)
                            else:
                                # print('save word-level', word_pic_nums[0],
                                # np.where(np.isnan(val[word_pic_nums[1] - 1].numpy())))
                                np.save(f, val[word_pic_nums[1] - 1].numpy(), allow_pickle=True)
                        elif isinstance(word_pic_nums[1], list):
                            if word_pic_nums[1][0] <= (len(val)-1):
                                # print('save word-level', word_pic_nums[0],
                                # np.where(np.isnan(val[word_pic_nums[1]].numpy())))
                                np.save(f, val[word_pic_nums[1][0]].numpy(), allow_pickle=True)
                            else:
                                # print('save word-level', word_pic_nums[0],
                                # np.where(np.isnan(val[word_pic_nums[1] - 1].numpy())))
                                np.save(f, val[word_pic_nums[1][0] - 1].numpy(), allow_pickle=True)

                    # except IndexError and len(word_pic_nums[1]) > 1:
                    #     # print('Index Error')
                    #     np.save(f, np.mean(val[word_pic_nums[1][0]:word_pic_nums[1][-2]].numpy(), axis=0),
                    #             allow_pickle=True)

    # for every sentence with n tokens, create one sentence tensor with averaged sentence tokens
    # for every sentence tensor create one tensor containing all sentence tensors for every layer
    def read_in_avg_enc_data(self, para_tr_o_te, data_name):
        folder_name = self.path_save_encs + para_tr_o_te + '/layers/' + data_name + '/'
        results_name = self.path_save_encs + para_tr_o_te + 'results/' + data_name + '/'
        layers_list = listdir(folder_name)
        # print('layers_list', layers_list)
        # print('self.path_save_encs', self.path_save_encs)
        # print('results_name', results_name)

        for layer in tqdm(layers_list):
            print('\n\nReading in all sentence embeddings & creating one np array...\n\n')

            filenames = natsorted(next(walk(folder_name + layer + '/'), (None, None, []))[2])
            first_enc_file = np.load(folder_name + layer + '/' + filenames.pop(0))
            collected_np_arr = np.mean(first_enc_file, axis=0)
            # first_token_arr = first_enc_file[0]

            for sent in tqdm(filenames):
                enc_sent = np.load(folder_name + layer + '/' + sent)
                avg_np_array = np.mean(enc_sent, axis=0)
                collected_np_arr = np.row_stack((collected_np_arr, avg_np_array))
                # first_token_arr = np.row_stack((first_token_arr, enc_sent[0]))

            # print('results_name', results_name)
            # save averaged np-array for all sentences
            with open(results_name + 'all_sent_avg_v_' + layer + '.npy', 'wb') as f:
                try:
                    np.save(f, collected_np_arr, allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

            # save np-array with first token for all sentences
            # with open(self.path_save_encs + para_tr_o_te + 'results_f_t/' + 'first_token_' + layer + '.npy', 'wb')
            # as f:
            #     try:
            #         np.save(f, first_token_arr, allow_pickle=True)
            #     except FileExistsError as error:
            #         # print(error)
            #         pass

    def read_in_word_level_make_matrix(self, para_tr_o_te, noise_type=None):
        print('\n\nReading in all word and sentence embeddings & creating one np matrix...\n\n')

        if para_tr_o_te == 'test' and self.config_dict['config'] == 'noise':
            data_name = 'noise/'
        elif para_tr_o_te == 'test' and self.config_dict['config'] != 'noise':
            data_name = 'clean/'
        else:
            data_name = ''

        folder_name = self.path_save_encs + para_tr_o_te + '/layers/' + data_name
        results_name = self.path_save_encs + para_tr_o_te + '/results/' + data_name
        layers_list = listdir(folder_name)
        # print('layers_list', layers_list)
        # print('self.path_save_encs', self.path_save_encs)

        for layer in tqdm(layers_list):
            # print('layer: ', layer)
            # print(folder_name + layer + '/')
            if self.config_dict['config'] == 'noise':
                name_folder_path = folder_name + layer + '/' + noise_type + '/'
                out_filename = results_name + noise_type + '/' + 'all_word_arrays_matrix_' + layer + '.npy'
            else:
                print('else', layer, '\n', folder_name)
                name_folder_path = folder_name + layer + '/'
                out_filename = results_name + 'all_word_arrays_matrix_' + layer + '.npy'
            print('name_folder_path', name_folder_path)
            filenames = natsorted(next(walk(name_folder_path), (None, None, []))[2])
            # print('filenames', filenames)
            first_name_file = filenames.pop(0)
            collected_np_arr_matrix = np.load(name_folder_path + first_name_file)
            # collected_np_label_array = np.array(first_name_file.split('.n')[0].split('_')[5])
            # collected_np_arr = np.mean(first_enc_file, axis=0)
            # first_token_arr = first_enc_file[0]

            for word_file in tqdm(filenames):
                enc_word = np.load(name_folder_path + word_file)
                # print('word_file: ', folder_name + layer + '/' + word_file, 'matrix, np.where: ', np.where(np.isnan(enc_word)))
                # enc_label = np.array(word_file.split('.n')[0].split('_')[5])
                collected_np_arr_matrix = np.row_stack((collected_np_arr_matrix, enc_word))
                # print('word_file: ', word_file, 'matrix, np.where: ', np.where(np.isnan(collected_np_arr_matrix)))
                # collected_np_label_array = np.append(collected_np_label_array, enc_label)
                # first_token_arr = np.row_stack((first_token_arr, enc_sent[0]))
            # if layer == 'l1':
            #     print('collected_np_label_array: ', collected_np_label_array)

            # print('results_name', results_name)
            # save averaged np-array for all sentences
            with open(out_filename, 'wb') as f:
                try:
                    # print('matrix, np.where: ', np.where(np.isnan(collected_np_arr_matrix)))
                    np.save(f, collected_np_arr_matrix, allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

            # with open(results_name + 'all_word_POS_array_' + layer + '.npy', 'wb') as f:
            #     try:
            #         # print('labels, np.where: ', len(np.where(np.isnan(collected_np_label_array))))
            #         np.save(f, collected_np_label_array, allow_pickle=True)
            #     except FileExistsError as error:
            #         # print(error)
            #         pass

    def create_shuffled_data_and_labels(self, data_features_dir, data_labels_file, size, data):
        train_test_dict = defaultdict()

        for file in data:
            layer = file.split('.')[0].split('_')[-1]

            # labels_all = np.load(self.path_save + data_labels_file, allow_pickle=True)
            labels_all = np.load(self.file_path + data_labels_file, allow_pickle=True)

            features_shuffled, labels_shuffled = shuffle(features, labels, random_state=42)

            train_test_dict[layer] = features_shuffled, labels_shuffled

        return train_test_dict

    def create_layer_stack_tensor(self, layer_list_dict):
        layer_stack_dict = defaultdict()

        for key_layer, val_avg_enc_list in layer_list_dict.items():
            layer_stack_dict[key_layer] = np.concatenate(val_avg_enc_list, axis=0)

        return layer_stack_dict

    # classify the sentence encoding for every layer
    def logistic_regression_classifier(self, v_or_t, train_feat_dir, train_labels, test_feat_dir, test_labels, size):
        filenames_train = natsorted(next(walk(self.path_save + train_feat_dir), (None, None, []))[2])
        filenames_test = natsorted(next(walk(self.path_save + test_feat_dir), (None, None, []))[2])

        train_dict = self.create_shuffled_data_and_labels(train_feat_dir, train_labels, size, filenames_train)
        test_dict = self.create_shuffled_data_and_labels(test_feat_dir, test_labels, size, filenames_test)

        collect_scores = defaultdict()
        collect_dummy_scores = defaultdict()

        for layer in sorted(train_dict.keys()):
            lr_clf = LogisticRegression(random_state=42)
            train_features, train_labels = train_dict[layer]
            test_features, test_labels = test_dict[layer]

            lr_clf.fit(train_features, train_labels)
            print('\n\n' + layer + '_lrf_score', lr_clf.score(test_features, test_labels))
            collect_scores[layer] = lr_clf.score(test_features, test_labels)

            dummy_clf = DummyClassifier()
            dummy_scores = cross_val_score(dummy_clf, train_features, train_labels)
            collect_dummy_scores[layer] = dummy_scores.mean()

            # save the model to disk
            filename = self.path_save + v_or_t + '/' + v_or_t + '_' + layer + '_lin_reg_model_' + str(size) + '.sav'
            pickle.dump(lr_clf, open(filename, 'wb'))

        return collect_scores, collect_dummy_scores

    def mlp_classifier(self, v_or_t, size, norm=None):
        print('Training MLP Classifier...')
        train_path = self.path_save_encs + 'train/results/'
        if self.config_dict['config'] == 'noise':
            test_path = self.path_save_encs + 'test/results/noise/'
        else:
            test_path = self.path_save_encs + 'test/results/clean/'
        filenames_train = natsorted(next(walk(train_path), (None, None, []))[2])
        tasks_dict = defaultdict()

        if norm:
            try:
                class_path = '/norm_mlp_sav/'
                os.mkdir(self.path_save_encs + class_path)
                class_path = '/norm_mlp_sav/norm_'
            except FileExistsError:
                pass
        else:
            try:
                class_path = '/mlp_sav/'
                os.mkdir(self.path_save_encs + class_path)
            except FileExistsError:
                pass

        if self.config_dict['tasks_word'][0] == 'dep':
            task_list = ['upos', 'xpos', 'dep']
        elif self.config_dict['tasks_word'][0] == 'sem':
            task_list = ['sem']

        for task in task_list:
            print('task: ', task)

            filenames_test = natsorted(next(walk(test_path), (None, None, []))[2])
            train_features = sorted(list(filter(lambda k: 'matrix' in k, filenames_train)))
            test_features = sorted(list(filter(lambda k: 'matrix' in k, filenames_test)))

            collect_scores = defaultdict()
            collect_dummy_scores = defaultdict()

            # with open('classification_report_TASK_' + task + '.txt', 'w', encoding='utf-8') as file:
            for train_feat, test_feat in list(zip(train_features, test_features)):

                layer = train_feat.split('.')[0].split('_')[-1]
                # print('layer', layer)
                train_features = np.load(train_path + train_feat, allow_pickle=True)
                train_labels = np.load(self.path_save_encs + 'train_' + task + '_all_labels_array.npy', allow_pickle=True)
                # print('train_labels: ', train_labels)
                test_features = np.load(test_path + test_feat, allow_pickle=True)
                test_labels = np.load(self.path_save_encs + 'test_' + task + '_all_labels_array.npy', allow_pickle=True)
                # print('train_labels', train_labels)
                # print(train_features.shape, train_labels.shape)
                # print(test_features.shape, test_labels.shape)
                # print(test_features[100])
                # print(np.where(np.isnan(test_features)))

                # print(test_labels[100])
                # print('train: ', len(train_features), len(train_labels))
                # print('test: ', len(test_features), len(test_labels))

                if norm:
                    sc = StandardScaler()
                    train_features = sc.fit_transform(train_features)
                    test_features = sc.transform(test_features)

                mlp_clf = MLPClassifier(random_state=1, max_iter=10000).fit(train_features, train_labels)
                #  print('mlp_clf.predict: ', mlp_clf.predict(test_features[:5, :]))
                # print('\n\n' + layer + '_mlp_score', mlp_clf.score(test_features, test_labels))
                collect_scores[layer] = mlp_clf.score(test_features, test_labels)
                # print('collect_scores', collect_scores)

                # use model to make predictions on test data
                # y_pred = mlp_clf.predict(test_features)

                # report = classification_report(test_labels, y_pred, output_dict=True)
                # report_df = pd.DataFrame(report)
                # print(report_df)
                # dfAsString = report_df.to_string(header=False, index=False)
                # file.write('Layer ' + layer + '\n\n')
                # file.write(dfAsString)

                # dummy_clf = DummyClassifier()
                # dummy_scores = cross_val_score(dummy_clf, train_features, train_labels)
                # collect_dummy_scores[layer] = dummy_scores.mean()

                # save the model to disk
                filename = self.path_save + v_or_t + class_path + task + '_' + self.config_dict['sent_word_prob'] + \
                           '_' + v_or_t + '_' + layer + '_mlp_model_10kit_' + str(size) + '.sav'
                pickle.dump(mlp_clf, open(filename, 'wb'))

            tasks_dict[task] = collect_scores
            print(tasks_dict)

        return tasks_dict

    def log_reg_no_dict_classifier(self, v_or_t, size, norm=None):
        print('Training log_reg Classifier...')
        train_path = self.path_save_encs + 'train/results/'
        if self.config_dict['config'] == 'noise':
            test_path = self.path_save_encs + 'test/results/noise/'
        else:
            test_path = self.path_save_encs + 'test/results/clean/'
        filenames_train = natsorted(next(walk(train_path), (None, None, []))[2])
        print('filenames_train: ', filenames_train, 'train_path: ',  train_path)
        # train_features = sorted(list(filter(lambda k: 'matrix' in k, filenames_train)))

        if norm:
            try:
                class_path = '/norm_lr_sav/'
                os.mkdir(self.path_save_encs + class_path)
                class_path = '/norm_lr_sav/norm_'
            except FileExistsError:
                pass
        else:
            try:
                class_path = '/lr_sav/'
                os.mkdir(self.path_save_encs + class_path)
            except FileExistsError:
                pass

        tasks_dict = defaultdict()

        if self.config_dict['tasks_word'][0] == 'dep':
            task_list = ['upos', 'xpos', 'dep']
        elif self.config_dict['tasks_word'][0] == 'sem':
            task_list = ['sem']

        for task in task_list:
        # for task in ['dep', 'xpos']:
            print('task: ', task)
            # print(train_features)
            # print(self.path_save_encs + 'train/results/')
            # filenames_test = natsorted(next(walk(test_path), (None, None, []))[2])
            # test_features = sorted(list(filter(lambda k: 'matrix' in k, filenames_test)))

            filenames_test = natsorted(next(walk(test_path), (None, None, []))[2])
            train_features = sorted(list(filter(lambda k: 'matrix' in k, filenames_train)))
            test_features = sorted(list(filter(lambda k: 'matrix' in k, filenames_test)))

            collect_scores = defaultdict()
            collect_dummy_scores = defaultdict()

            for train_feat, test_feat in list(zip(train_features, test_features)):
                layer = train_feat.split('.')[0].split('_')[-1]
                print('layer', layer)
                train_features = np.load(train_path + train_feat, allow_pickle=True)
                train_labels = np.load(self.path_save_encs + 'train_' + task + '_all_labels_array.npy', allow_pickle=True)
                test_features = np.load(test_path + test_feat, allow_pickle=True)
                test_labels = np.load(self.path_save_encs + 'test_' + task + '_all_labels_array.npy', allow_pickle=True)
                # print('train_labels', train_labels)

                if norm:
                    sc = StandardScaler()
                    train_features = sc.fit_transform(train_features)
                    test_features = sc.transform(test_features)

                lr_clf = LogisticRegression(random_state=42, max_iter=10000).fit(train_features, train_labels)

                # print('\n\n' + layer + '_lr_score', lr_clf.score(test_features, test_labels))
                collect_scores[layer] = lr_clf.score(test_features, test_labels)
                # print('collect_scores', collect_scores)

                # use model to make predictions on test data
                y_pred = lr_clf.predict(test_features)
                # print(test_labels, y_pred)
                # print(classification_report(test_labels, y_pred))

                dummy_clf = DummyClassifier()
                dummy_scores = cross_val_score(dummy_clf, train_features, train_labels)
                collect_dummy_scores[layer] = dummy_scores.mean()

                # save the model to disk
                filename = self.path_save + v_or_t + class_path + task + '_' + self.config_dict['sent_word_prob'] + \
                           '_' + v_or_t + '_' + layer + '_lr_model_10kit_' + str(size) + '.sav'
                pickle.dump(lr_clf, open(filename, 'wb'))

            tasks_dict[task] = collect_scores
            print(tasks_dict)

        return tasks_dict

    def process_raw_test_data_forward_to_saved_classifier(self, size):
        pass
        df_test = self.read_in_avg_enc_data('test', data_name)
        df_labels = df_test[1].to_numpy()

        # TODO !!!
        # layers_avg_enc_list_dict, all_layers_sent_list_dict = self.translate(df_test, None, False)
        layers_avg_enc_list_dict, all_layers_sent_list_dict = self.translate_process(df_test)
        stack_layer_dict = self.create_layer_stack_tensor(layers_avg_enc_list_dict)
        collect_scores = self.load_classifier_model()

    def load_classifier_model_load_avg_encs(self, path_avg_encs, path_classifier, path_labels):
        print('Loading trained classifier for sentence-level evaluation...')

        classifier_list = natsorted(next(walk(path_classifier), (None, None, []))[2])
        eval_files_list = natsorted(next(walk(path_avg_encs), (None, None, []))[2])
        layer_list = [l.split('.')[0].split('_')[-1] for l in eval_files_list]
        df_labels = np.load(path_labels, allow_pickle=True)
        collect_scores = defaultdict()

        for layer in layer_list:
            # load the model from disk
            classifier_model = path_classifier + [elem for elem in classifier_list if layer in elem][0]
            eval_file = np.load(path_avg_encs + [elem for elem in eval_files_list if layer in elem][0],
                                allow_pickle=True)
            loaded_model = pickle.load(open(classifier_model, 'rb'))
            test_features, test_labels = shuffle(eval_file, df_labels, random_state=42)

            collect_scores[layer] = loaded_model.score(test_features, test_labels)

        # return df_test, collect_scores
        return collect_scores

    def load_classifier_model_word_level(self, w_task, path_avg_encs, path_classifier,
                                         path_labels):  # , path_out_class_report=None):
        print('Loading trained classifier for word-level evaluation...')

        classifier_list = natsorted(next(walk(path_classifier), (None, None, []))[2])
        print('path_classifier: ', path_classifier)
        eval_files_list = natsorted(filter(lambda k: 'matrix' in k, next(walk(path_avg_encs), (None, None, []))[2]))
        layer_list = [l.split('.')[0].split('_')[-1] for l in eval_files_list]
        collect_scores = defaultdict()
        f1_collect_scores = defaultdict()
        df_labels = np.load(path_labels, allow_pickle=True)

        # with open(path_out_class_report, 'w') as out:
            # out.write('Classification Report\n\n')
        for layer in layer_list:
            # load the model from disk
            # out.write('\nLayer ' + layer[-1] + '\n')
            filtered_class_list = list(filter(lambda cla: w_task in cla, classifier_list))
            classifier_model = path_classifier + [elem for elem in filtered_class_list if layer in elem][0]
            eval_file = np.load(path_avg_encs + [elem for elem in eval_files_list if layer in elem][0],
                                allow_pickle=True)
            loaded_model = pickle.load(open(classifier_model, 'rb'))
            test_features, test_labels = shuffle(eval_file, df_labels, random_state=42)

            # use model to make predictions on test data
            y_pred = loaded_model.predict(test_features)
            class_rep = classification_report(test_labels, y_pred)
            # out.write(class_rep)

            collect_scores[layer] = loaded_model.score(test_features, test_labels)
            f1_collect_scores[layer] = f1_score(test_labels, y_pred, average='macro')
            # collect_scores[layer] = loaded_model.score(test_features, test_labels)
            # print(layer, balanced_accuracy_score(test_labels, y_pred))
            print(layer, f1_score(test_labels, y_pred, average='macro'))
            print(layer, loaded_model.score(test_features, test_labels))

        # print(collect_scores)

        # return df_test, collect_scores
        return collect_scores, f1_collect_scores

    # ToDo
    def sanity_check(self, data_features_dir):
        filenames = natsorted(next(walk(self.path_save_encs + data_features_dir), (None, None, []))[2])
        features_l1 = np.load(self.path_save_encs + data_features_dir + filenames[0], allow_pickle=True)
        features_l2 = np.load(self.path_save_encs + data_features_dir + filenames[1], allow_pickle=True)
        features_l3 = np.load(self.path_save_encs + data_features_dir + filenames[2], allow_pickle=True)
        features_l4 = np.load(self.path_save_encs + data_features_dir + filenames[3], allow_pickle=True)
        features_l5 = np.load(self.path_save_encs + data_features_dir + filenames[4], allow_pickle=True)
        features_l6 = np.load(self.path_save_encs + data_features_dir + filenames[5], allow_pickle=True)

        test_l1_l2 = features_l1 == features_l2
        test_l1_l3 = features_l1 == features_l3
        test_l1_l4 = features_l1 == features_l4
        test_l1_l5 = features_l1 == features_l5
        test_l1_l6 = features_l1 == features_l6
        test_l2_l3 = features_l2 == features_l3
        test_l2_l4 = features_l2 == features_l4
        test_l2_l5 = features_l2 == features_l5
        test_l2_l6 = features_l2 == features_l6
        test_l3_l4 = features_l3 == features_l4
        test_l3_l5 = features_l3 == features_l5
        test_l3_l6 = features_l3 == features_l6
        test_l4_l5 = features_l4 == features_l5
        test_l4_l6 = features_l4 == features_l6
        test_l5_l6 = features_l5 == features_l6

        print(test_l1_l2, test_l1_l3, test_l1_l4, test_l1_l5, test_l1_l6, test_l2_l3, test_l2_l4, test_l2_l5,
              test_l2_l6, test_l3_l4, test_l3_l5, test_l3_l6, test_l4_l5, test_l4_l6, test_l5_l6)
