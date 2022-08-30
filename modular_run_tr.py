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
import os
from os import walk, listdir
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
import pickle


class VisRepEncodings:
    def __init__(self, file, path_save_encs):
        self.model = None
        self.encoded_data = None
        self.data = file
        self.train_dict = defaultdict
        self.m_para = None
        self.path_save = path_save_encs
        self.path_save_encs = None
        self.create_layer_path = True
        self.single_layer = 'l6'
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

    def read_in_raw_data(self, d_size, train_or_test_size, test_train):
        print('\nRead in raw data...\n')

        half_data_size = int(d_size*train_or_test_size / 2)
        df_all = pd.read_csv(self.data, delimiter='\t', header=None)
        df_values = set(df_all[1].values)

        df_combined = pd.DataFrame(columns=df_all.columns.values)

        for val in df_values:
            df_val = df_all[df_all[1] == val].sample(half_data_size)
            df_combined = pd.concat([df_combined, df_val], axis=0)

        df_combined.reset_index(inplace=True, drop=True)

        print('df_combined', df_combined[1].value_counts(), len(df_combined))

        with open(str(self.path_save) + test_train + '_raw_sentences.npy', 'wb') as f:
            try:
                np.save(f, np.array(df_combined[0]), allow_pickle=True)
            except FileExistsError as error:
                print(error)
                # pass
        
        with open(str(self.path_save) + test_train + '_raw_labels.npy', 'wb') as f:
            try:
                np.save(f, np.array(df_combined[1]), allow_pickle=True)
            except FileExistsError as error:
                print(error)
                # pass

        return df_combined

    # Translate sentences
    def translate(self, batch, tr_or_te):
        print('\n\nTranslating sentences // Creating encodings...\n\n')

        make_directories = True

        for idx, sent in tqdm(enumerate(batch[0])):
            translation, layer_dict = self.model.translate(sent)
            print(translation)

            if make_directories:
                make_directories = False
                self.make_directories(layer_dict, tr_or_te)

            self.save_encodings(layer_dict, idx, tr_or_te)

    # Create directories for layers
    def make_directories(self, np_dict, tr_or_te):
        self.all_layers = np_dict.keys()

        try:
            os.mkdir(self.path_save_encs + tr_or_te + '/')
        except OSError as error:
            # print(error)
            pass

        try:
            os.mkdir(self.path_save_encs + tr_or_te + '/layers/')
        except OSError as error:
            # print(error)
            pass

        if self.create_layer_path:
            for key in np_dict.keys():
                try:
                    os.mkdir(self.path_save_encs + tr_or_te + '/layers/' + key + "/")
                except OSError as error:
                    # print(error)
                    continue

    def save_encodings(self, np_dict, sent_num, tr_or_te):
        data_name = self.data.split('/')[1].split('.')[0]

        for key, val in np_dict.items():
            with open(self.path_save_encs + tr_or_te + '/layers/' + key + '/' + data_name + '_' + self.m_para + '_sent_'
                      + str(sent_num) + '.npy', 'wb') as f:
                try:
                    np.save(f, val.numpy(), allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

    # for every sentence with n tokens, create one sentence tensor with averaged sentence tokens
    # for every sentence tensor create one tensor containing all sentence tensors for every layer
    def read_in_avg_enc_data(self, para_tr_o_te):
        layers_list = listdir(self.path_save_encs + para_tr_o_te + 'layers/')

        for layer in tqdm(layers_list):
            print('\n\nCollecting all sentence embeddings & creating one np array...\n\n')

            filenames = natsorted(next(walk(self.path_save_encs + para_tr_o_te + 'layers/' + layer + '/'), (None, None, []))[2])
            first_enc_file = np.load(self.path_save_encs + para_tr_o_te + 'layers/' + layer + '/' + filenames.pop(0))
            collected_np_arr = np.sum(first_enc_file, axis=0) / first_enc_file.shape[0]
            first_token_arr = first_enc_file[0]

            for file in tqdm(filenames):
                enc_file = np.load(self.path_save_encs + para_tr_o_te + 'layers/' + layer + '/' + file)
                avg_np_array = np.mean(enc_file, axis=0)
                collected_np_arr = np.row_stack((collected_np_arr, avg_np_array))
                first_token_arr = np.row_stack((first_token_arr, enc_file[0]))

            for res_path in ('results/', 'results_f_t/'):
                try:
                    os.mkdir(self.path_save_encs + para_tr_o_te + res_path)
                except OSError as error:
                    # print(error)
                    continue

            # save averaged np-array for all sentences
            with open(self.path_save_encs + para_tr_o_te + 'results/' + 'all_sent_avg_v_' + layer + '.npy', 'wb') as f:
                try:
                    np.save(f, collected_np_arr, allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

            # save np-array with first token for all sentences
            with open(self.path_save_encs + para_tr_o_te + 'results_f_t/' + 'first_token_' + layer + '.npy', 'wb') as f:
                try:
                    np.save(f, collected_np_arr, allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

    def create_shuffled_data_and_labels(self, data_features_dir, data_labels_file, size, data):
        train_test_dict = defaultdict()

        for file in data:
            half_size = int(size / 2)
            layer = file.split('.')[0].split('_')[-1]
            features_all = np.load(self.path_save + data_features_dir + file, allow_pickle=True)
            features = np.concatenate((features_all[:half_size], features_all[-half_size:]), axis=0)

            labels_all = np.load(self.path_save + data_labels_file, allow_pickle=True)
            labels = np.concatenate((labels_all[:half_size], labels_all[-half_size:]), axis=0)

            features_shuffled, labels_shuffled = shuffle(features, labels, random_state=42)

            train_test_dict[layer] = features_shuffled, labels_shuffled

        return train_test_dict

    # classify the sentence encoding for every layer
    def logistic_regression_classifier(self, train_feat_dir, train_labels, test_feat_dir, test_labels, size):
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
            filename = self.path_save_encs + layer + '_lin_reg_model.sav'
            pickle.dump(lr_clf, open(filename, 'wb'))

        return collect_scores, collect_dummy_scores

    def load_classifier_model(self, test_feat_dir, test_labels, size):
        filenames_test = natsorted(next(walk(self.path_save + test_feat_dir), (None, None, []))[2])
        test_dict = self.create_shuffled_data_and_labels(test_feat_dir, test_labels, size, filenames_test)

        collect_scores = defaultdict()

        for layer in sorted(test_dict.keys()):
            # load the model from disk
            filename = self.path_save_encs + layer + '_lin_reg_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            test_features, test_labels = test_dict[layer]

            collect_scores[layer] = loaded_model.score(test_features, test_labels)

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
