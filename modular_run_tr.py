# Load the model in python
from fairseq.models.visual import VisualTextTransformerModel
from fairseq.models.transformer import TransformerModel
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
import os
from os import walk, listdir
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt


class VisRepEncodings:
    def __init__(self, file, single_layer, path_save_encs):
        self.model = None
        self.encoded_data = None
        self.data = file
        self.train_dict = defaultdict
        self.m_para = None
        self.path_save = path_save_encs
        self.path_save_encs = None
        self.create_layer_path = True
        self.single_layer = single_layer
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
        print(self.path_save_encs)

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
        print(self.path_save_encs)

        try:
            os.mkdir(self.path_save_encs)
        except OSError as error:
            # print(error)
            pass

    def read_in_raw_data(self, data_size):
        df = pd.read_csv(self.data, delimiter='\t', header=None)[:data_size]
        # batch_1 = df[:2000]
        # print(batch_1[1].value_counts(), len(batch_1[0]))
        print(df[1].value_counts(), len(df[0]))

        print(self.path_save_encs)

        with open(self.path_save_encs + 'raw_sentences.npy', 'wb') as f:
            try:
                np.save(f, np.array(df[0]), allow_pickle=True)
            except FileExistsError as error:
                # print(error)
                pass
        
        print(np.array(df[1]))

        with open(self.path_save_encs + 'raw_labels.npy', 'wb') as f:
            try:
                np.save(f, np.array(df[1]), allow_pickle=True)
            except FileExistsError as error:
                # print(error)
                pass

        return df

    # TODO !
    def read_in_avg_enc_data(self, para_avg):
        # pass

        layers_list = listdir(self.path_save_encs + 'layers/')

        for layer_dir in tqdm(layers_list):

            print('\n\nCollecting all sentence embeddings & creating one np array...\n\n')

            filenames = natsorted(next(walk(self.path_save_encs + 'layers/' + layer_dir + '/'), (None, None, []))[2])
            first_enc_file = np.load(self.path_save_encs + 'layers/' + layer_dir + '/' + filenames.pop(0))
            collected_np_arr = np.sum(first_enc_file, axis=0) / first_enc_file.shape[0]
            first_token_arr = first_enc_file[0]

            for file in tqdm(filenames):
                if para_avg:
                    enc_file = np.load(self.path_save_encs + 'layers/' + self.single_layer + '/' + file)
                    avg_np_array = np.sum(enc_file, axis=0) / enc_file.shape[0]
                    collected_np_arr = np.row_stack((collected_np_arr, avg_np_array))
                    first_token_arr = np.row_stack((first_token_arr, enc_file[0]))

            for res_path in ('results/', 'results_f_t/'):
                try:
                    os.mkdir(self.path_save_encs + res_path)
                except OSError as error:
                    # print(error)
                    continue

            # save averaged np-array for all sentences
            with open(self.path_save_encs + 'results/' + 'all_sent_avg_v_' + layer_dir + '.npy', 'wb') as f:
                try:
                    np.save(f, collected_np_arr, allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

            # save np-array with first token for all sentences
            with open(self.path_save_encs + 'results_f_t/' + 'first_token_' + layer_dir + '.npy', 'wb') as f:
                try:
                    np.save(f, collected_np_arr, allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

    def make_dir_save_enc(self, np_dict, sent_num, create_para):
        self.create_layer_path = create_para
        path = self.path_save_encs
        data_name = self.data.split('/')[1].split('.')[0]
        self.all_layers = np_dict.keys()

        # print(np_dict.items())
        
        try:
            os.mkdir(path + 'layers/')
        except OSError as error:
            # print(error)
            pass

        if self.create_layer_path:
            for key in np_dict.keys():
                try:
                    os.mkdir(path + 'layers/' + key + "/")
                except OSError as error:
                    # print(error)
                    continue

        self.create_layer_path = False

        for key, val in np_dict.items():
            with open(path + 'layers/' + key + '/' + data_name + '_' + self.m_para + '_sent_' + str(sent_num) + '.npy', 'wb') as f:
                try:
                    np.save(f, val.numpy(), allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

    # TODO !
    def pad_encoded_data(self, data):
        # example
        a = np.array([1, 2, 3, 4])
        b = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        a.resize(b.shape)
        # >> > a
        # array([1, 2, 3, 4, 0, 0, 0, 0])

    def logistic_regression_classifier(self, data_features_dir, data_labels_file, size):
        collect_scores = defaultdict()

        filenames = natsorted(next(walk(self.path_save_encs + data_features_dir), (None, None, []))[2])
        features_shape = None

        for file in filenames:
            file_name = file.split('.')[0].split('_')[-1]
            features = np.load(self.path_save_encs + data_features_dir + file, allow_pickle=True)
            features_shape = features.shape[0]
            labels = np.load(self.path_save_encs + data_labels_file, allow_pickle=True)

            train_features, test_features, train_labels, test_labels = train_test_split(features[:size], labels[:size])
            # print(len(train_labels), len(test_labels))

            lr_clf = LogisticRegression()
            lr_clf.fit(train_features, train_labels)

            print('\n\n' + file_name + '_lrf_score', lr_clf.score(test_features, test_labels))
            collect_scores[file_name] = lr_clf.score(test_features, test_labels)

        return collect_scores

        # self.plot_results(collect_scores, data_features_dir.split('/')[0], size)

    def plot_results(self, data_dict, data_one_t_dict, size):
        # data_dict = {'l1': 0.3, 'l2': 0.4, 'l3': 0.5, 'l4': 0.6, 'l5': 0.7, 'l6': 0.8, 'l6_ln': 0.9}
        # print(data_dict.keys(), data_dict.values())

        a = plt.scatter(data_dict.keys(), data_dict.values())
        b = plt.scatter(data_one_t_dict.keys(), data_one_t_dict.values())
        plt.xlabel('Layers')
        plt.ylabel('Results Linear Classifier')

        plt.legend([a,b], ['Averaged tokens', 'First token'])

        # plt.legend(data_feat_dir)

        # plt.show()

        plt.savefig(self.path_save_encs + 'results_' + self.path_save_encs.split('/')[5] + '_' + str(size) + '.png')#, bbox_inches='tight')
        plt.close()

    # Translate
    def translate(self, batch):
        print('\n\nTranslating sentences // Creating encodings...\n\n')
        
        for idx, sent in tqdm(enumerate(batch[0])):
            translation, layer_dict = self.model.translate(sent)
            # print(translation)
            # print(layer_dict.keys())
            # break
            self.make_dir_save_enc(layer_dict, idx, True)


if __name__ == '__main__':
    # local
    RunVisrep_TENSE = VisRepEncodings('/home/anastasia/PycharmProjects/xprobe/de/past_present/tense.txt', 'l6',
                                      '/home/anastasia/PycharmProjects/visrepProb/task_encs/past_pres/')

    RunVisrep_OBJ = VisRepEncodings('/home/anastasia/PycharmProjects/xprobe/de/past_present/tense.txt', 'l6',
                                    '/home/anastasia/PycharmProjects/visrepProb/task_encs/past_pres/')

    # server
    RunVisrep_TENSE = VisRepEncodings('/local/anasbori/xprobe/de/past_present/tense.txt',
                                'l6',
                                '/local/anasbori/visrepProb/task_encs/past_present/')

    # RunVisrep_OBJ = VisRepEncodings('/local/anasbori/xprobe/de/obj_number/objnum.txt',
    #                             'l6',
    #                             '/local/anasbori/visrepProb/task_encs/obj_number/')

    RunVisrep_SUBJ = VisRepEncodings('/local/anasbori/xprobe/de/subj_number/subjnum.txt',
                                'l6',
                                '/local/anasbori/visrepProb/task_encs/subj_number/')

    # RunVisrep_BIGRAM = VisRepEncodings('/local/anasbori/xprobe/de/bigram_shift/bishift.txt',
    #                             'l6',
    #                             '/local/anasbori/visrepProb/task_encs/bigram_shift/')

    # for RunVisrep in (RunVisrep_OBJ, RunVisrep_BIGRAM):
    for RunVisrep in (RunVisrep_TENSE, RunVisrep_SUBJ):

        for m_type in ('v', 't'):
            RunVisrep.make_vis_model(m_type)

            RunVisrep.translate(RunVisrep.read_in_raw_data(10000))
            RunVisrep.read_in_avg_enc_data(True)

            for data_size in (10000, 1000):
                results = RunVisrep.logistic_regression_classifier('results/', 'raw_labels.npy', data_size)
                results_f_t = RunVisrep.logistic_regression_classifier('results_f_t/', 'raw_labels.npy', data_size)
                RunVisrep.plot_results(results, results_f_t, data_size)
