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
import os
from os import walk, listdir
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
import pickle
import yaml
from get_image_slices_clean import get_wordpixels_in_pic_slice, get_pic_num_for_word


class VisRepEncodings:
    def __init__(self, config_dict, file_path, path_save_encs):
        self.config_dict = config_dict
        self.model = None
        self.encoded_data = None
        self.file_path = file_path
        # if self.config_dict['config'] == 'noise':
        # self.data = file_path + self.config_dict['noise_test_file_out']
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
        print(str(self.path_save) + test_train + '_raw_sentences.npy')
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

    def translate_word_level_save(self, batch, tr_or_te, data_name):
        print('\n\nTranslating sentences // Creating encodings...\n\n')

        make_directories = True

        for idx, sent in tqdm(enumerate(batch)):
            sent_list, pos_list = list(zip(*sent))
            translation, layer_dict = self.model.translate(' '.join(sent_list))
            # print('len(layer_dict[l1])', layer_dict['l1'].shape)
            # for key, item in layer_dict.items():
            #     print('np.where: ', np.where(np.isnan(layer_dict[key])))
            pic_num_words = get_pic_num_for_word(get_wordpixels_in_pic_slice(' '.join(sent_list)))
            # print('translation', translation)

            if make_directories:
                print('Make directories...')
                make_directories = False
                self.make_directories(layer_dict)

            self.save_word_level_encodings(layer_dict, idx, tr_or_te, data_name, pic_num_words, pos_list)

    def translate_process(self, batch):
        collect_layer_dicts = defaultdict(list)
        collect_idx_sent_dict = defaultdict(list)

        for idx, sent in tqdm(enumerate(batch[0])):
            print(sent)
            translation, layer_dict = self.model.translate(sent)
            for key_layer, val_enc in layer_dict.items():
                collect_layer_dicts[key_layer].append(np.mean(val_enc.numpy(), axis=0))
                collect_idx_sent_dict[key_layer].append(sent)

        return collect_layer_dicts, collect_idx_sent_dict

    # Create directories for layers
    def make_directories(self, np_dict):
        print('self.path_save_encs', self.path_save_encs)
        self.all_layers = np_dict.keys()
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

        if self.create_layer_path:
            for key in np_dict.keys():
                try:
                    os.mkdir(self.path_save_encs + 'train/layers/' + key + '/')
                except OSError as error:
                    # print(error)
                    continue

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

    def save_word_level_encodings(self, np_dict, sent_num, tr_or_te, d_name, pic_num_words_list, pos_list):
        # data_name = self.data.split('/')[1].split('.')[0]
        # print(data_name)
        # print('Saving encodings...')
        for key, val in np_dict.items():
            np.array(val)
            # print('val: \n', val[0:3])
            # print(np.mean(key_val[1][0:3].numpy(), axis=0))
            # np.save(f, np.mean(val[pic_nums[0]:pic_nums[-1]]), allow_pickle=True)
            count_val = -1
            for word_pic_nums, pos in list(zip(pic_num_words_list, pos_list)):
                # print('pic_nums', pic_nums)
                count_val += 1
                if tr_or_te == 'test' and self.config_dict['config'] == 'noise':
                    path_tr_te = 'noise/'
                elif tr_or_te == 'test' and self.config_dict['config'] != 'noise':
                    path_tr_te = 'clean/'
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
                with open(self.path_save_encs + tr_or_te + '/layers/' + path_tr_te + key + '/' + d_name
                          + '_' + self.m_para + '_sent' + str(sent_num) + '_word' + str(count_val) + '_' +
                          word_file + '_' + pos + '.npy', 'wb') as f:
                    # print('word_pic_nums[1]: ', word_pic_nums[1], 'len(val): ', len(val))
                    try:
                        # print('Normal')
                        # print(np.mean(val[word_pic_nums[1][0]:word_pic_nums[1][-1]]))
                        # print(len(word_pic_nums[1]))
                        if len(word_pic_nums[1]) == 1:
                            # print('len(word_pic_nums) == 1', word_pic_nums[1][0])
                            # print('save word-level', word_pic_nums[0], 'len(w_pic_nums): ', len(word_pic_nums[1]),
                            #      np.where(np.isnan(val[word_pic_nums[1][0]].numpy())))
                            np.save(f, val[word_pic_nums[1]].numpy(), axis=0, allow_pickle=True)
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
        print('layers_list', layers_list)
        print('self.path_save_encs', self.path_save_encs)
        print('results_name', results_name)

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

            print('results_name', results_name)
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

    def read_in_word_level_make_matrix(self, para_tr_o_te):
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
        print('layers_list', layers_list)
        print('self.path_save_encs', self.path_save_encs)
        print('results_name', results_name)

        for layer in tqdm(layers_list):

            filenames = natsorted(next(walk(folder_name + layer + '/'), (None, None, []))[2])
            # print('filenames', filenames)
            first_name_file = filenames.pop(0)
            collected_np_arr_matrix = np.load(folder_name + layer + '/' + first_name_file)
            collected_np_label_array = np.array(first_name_file.split('.n')[0].split('_')[5])
            # collected_np_arr = np.mean(first_enc_file, axis=0)
            # first_token_arr = first_enc_file[0]

            for word_file in tqdm(filenames):
                enc_word = np.load(folder_name + layer + '/' + word_file)
                # print('word_file: ', folder_name + layer + '/' + word_file, 'matrix, np.where: ', np.where(np.isnan(enc_word)))
                enc_label = np.array(word_file.split('.n')[0].split('_')[5])
                collected_np_arr_matrix = np.row_stack((collected_np_arr_matrix, enc_word))
                # print('word_file: ', word_file, 'matrix, np.where: ', np.where(np.isnan(collected_np_arr_matrix)))
                collected_np_label_array = np.append(collected_np_label_array, enc_label)
                # first_token_arr = np.row_stack((first_token_arr, enc_sent[0]))

            print('results_name', results_name)
            # save averaged np-array for all sentences
            with open(results_name + 'all_word_arrays_matrix_' + layer + '.npy', 'wb') as f:
                try:
                    # print('matrix, np.where: ', np.where(np.isnan(collected_np_arr_matrix)))
                    np.save(f, collected_np_arr_matrix, allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

            with open(results_name + 'all_word_POS_array_' + layer + '.npy', 'wb') as f:
                try:
                    # print('labels, np.where: ', len(np.where(np.isnan(collected_np_label_array))))
                    np.save(f, collected_np_label_array, allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

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

    # TODO: classify word-level encodings for every layer
    def mlp_classifier(self, v_or_t, train_feat_dir, train_labels, test_feat_dir, test_labels, size):
        print('Training MLP Classifier...')
        train_path = self.path_save_encs + 'train/results/'
        test_path = self.path_save_encs + 'test/results/clean/'
        filenames_train = natsorted(next(walk(train_path), (None, None, []))[2])
        train_features = sorted(list(filter(lambda k: 'matrix' in k, filenames_train)))
        train_labels = sorted(list(filter(lambda k: 'POS' in k, filenames_train)))
        # print(train_features)
        # print(self.path_save_encs + 'train/results/')
        filenames_test = natsorted(next(walk(test_path), (None, None, []))[2])
        test_features = sorted(list(filter(lambda k: 'matrix' in k, filenames_test)))
        test_labels = sorted(list(filter(lambda k: 'POS' in k, filenames_test)))

        collect_scores = defaultdict()
        collect_dummy_scores = defaultdict()

        for train_feat, train_lab, test_feat, test_lab in list(zip(train_features, train_labels, test_features,
                                                                   test_labels)):
            layer = train_feat.split('.')[0].split('_')[-1]
            print('layer', layer)
            train_features = np.load(train_path + train_feat, allow_pickle=True)
            train_labels = np.load(train_path + train_lab, allow_pickle=True)
            test_features = np.load(test_path + test_feat, allow_pickle=True)
            test_labels = np.load(test_path + test_lab, allow_pickle=True)
            print(train_features.shape, train_labels.shape)
            print(test_features.shape, test_labels.shape)
            # print(test_features[100])
            # print(np.where(np.isnan(test_features)))

            # print(test_labels[100])

            mlp_clf = MLPClassifier(random_state=1, max_iter=300).fit(train_features, train_labels)

            print('\n\n' + layer + '_lrf_score', mlp_clf.score(test_features, test_labels))
            collect_scores[layer] = mlp_clf.score(test_features, test_labels)
            print('collect_scores', collect_scores)

            dummy_clf = DummyClassifier()
            dummy_scores = cross_val_score(dummy_clf, train_features, train_labels)
            collect_dummy_scores[layer] = dummy_scores.mean()

            # save the model to disk
            filename = self.path_save + v_or_t + '/' + v_or_t + '_' + layer + '_mlp_model_' + str(size) + '.sav'
            pickle.dump(mlp_clf, open(filename, 'wb'))

        return collect_scores, collect_dummy_scores

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
        print('Loading trained classifier...')

        classifier_list = natsorted(next(walk(path_classifier), (None, None, []))[2])
        eval_files_list = natsorted(next(walk(path_avg_encs), (None, None, []))[2])
        layer_list = [l.split('.')[0].split('_')[-1] for l in eval_files_list]
        df_labels = np.load(path_labels,allow_pickle=True)
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
