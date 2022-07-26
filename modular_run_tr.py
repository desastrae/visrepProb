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
import sys


def plot_results_avg_f_t(enc_task, path_save, m_para, data_dict_all, size):
    a = plt.plot(data_dict_all['avg'].keys(), data_dict_all['avg'].values)
    b = plt.plot(data_dict_all['f_t'].keys(), data_dict_all['f_t'].values)
    plt.xlabel('Layers')
    plt.ylabel('Results Linear Classifier')

    plt.legend(['Averaged tokens', 'First token'])

    if m_para == 'v':
        plt.title('Visual model, task ' + enc_task + ', data-set size ' + str(size))
    else:
        plt.title('Text model, task ' + enc_task + ', data-set size ' + str(size))

    plt.savefig(path_save + m_para + '/' + m_para + '_results_' + enc_task + '_' + str(size) + '.png')
    plt.close()


def plot_results_v_vs_t(enc_task, path_save, data_dict_v, data_dict_t, size):
    # wasn't able to extract normalized Layer for text-model; temporary fix, drop normalized layer for v_model
    # data_dict_v.drop(data_dict_v.tail(1).index, inplace=True)

    labels = list(data_dict_v['avg'].keys())
    v_results = data_dict_v['avg'].values
    t_results = data_dict_t['avg'].values
    dummy_val_t = np.array(data_dict_t['dummy'].values).mean()
    # dummy_val_v = np.array(data_dict_v['dummy'].values).mean()

    # adding 0.0 for l7 to text-model data, otherwise I'd need more time to fix this
    t_results = np.append(t_results, 0.0)
    print(t_results)

    print('labels', labels, '\nv_results', v_results, '\nt_results', t_results)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, v_results, width, label='Visual Model')
    rects2 = ax.bar(x + width / 2, t_results, width, label='Text Model')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Results Linear Classifier')
    ax.set_title('Visual vs. Text model, task ' + enc_task + ', data-set size ' + str(size))
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.ylim([0.45, 1.0])

    plt.axhline(y=dummy_val_t, color='r', linestyle='-')

    plt.show()
    # plt.savefig(path_save + 'v_vs_t_results_' + enc_task + '_' + str(size) + '.png')
    plt.close()


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

    def read_in_avg_enc_data(self, para_avg):
        layers_list = listdir(self.path_save_encs + 'layers/')

        for layer in tqdm(layers_list):
            print('\n\nCollecting all sentence embeddings & creating one np array...\n\n')

            filenames = natsorted(next(walk(self.path_save_encs + 'layers/' + layer + '/'), (None, None, []))[2])
            first_enc_file = np.load(self.path_save_encs + 'layers/' + layer + '/' + filenames.pop(0))
            collected_np_arr = np.sum(first_enc_file, axis=0) / first_enc_file.shape[0]
            first_token_arr = first_enc_file[0]

            for file in tqdm(filenames):
                enc_file = np.load(self.path_save_encs + 'layers/' + layer + '/' + file)
                avg_np_array = np.mean(enc_file, axis=0)
                collected_np_arr = np.row_stack((collected_np_arr, avg_np_array))
                first_token_arr = np.row_stack((first_token_arr, enc_file[0]))

            for res_path in ('results/', 'results_f_t/'):
                try:
                    os.mkdir(self.path_save_encs + res_path)
                except OSError as error:
                    # print(error)
                    continue

            # save averaged np-array for all sentences
            with open(self.path_save_encs + 'results/' + 'all_sent_avg_v_' + layer + '.npy', 'wb') as f:
                try:
                    np.save(f, collected_np_arr, allow_pickle=True)
                except FileExistsError as error:
                    # print(error)
                    pass

            # save np-array with first token for all sentences
            with open(self.path_save_encs + 'results_f_t/' + 'first_token_' + layer + '.npy', 'wb') as f:
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
        collect_dummy_scores = defaultdict()

        filenames = natsorted(next(walk(self.path_save_encs + data_features_dir), (None, None, []))[2])
        features_shape = None

        for file in filenames:
            layer = file.split('.')[0].split('_')[-1]
            features = np.load(self.path_save_encs + data_features_dir + file, allow_pickle=True)
            features_shape = features.shape[0]
            labels = np.load(self.path_save_encs + data_labels_file, allow_pickle=True)

            train_features, test_features, train_labels, test_labels = train_test_split(features[:size], labels[:size])
            # print(len(train_labels), len(test_labels))

            lr_clf = LogisticRegression(random_state=42)
            lr_clf.fit(train_features, train_labels)

            print('\n\n' + layer + '_lrf_score', lr_clf.score(test_features, test_labels))
            collect_scores[layer] = lr_clf.score(test_features, test_labels)

            dummy_clf = DummyClassifier()
            dummy_scores = cross_val_score(dummy_clf, train_features, train_labels)

            collect_dummy_scores[layer] = dummy_scores.mean()

            print("Dummy classifier score for layer" + layer + ": %0.3f (+/- %0.2f)" % (dummy_scores.mean(),
                                                                                        dummy_scores.std() * 2))

        return collect_scores, collect_dummy_scores

    # Translate
    def translate(self, batch):
        print('\n\nTranslating sentences // Creating encodings...\n\n')
        
        for idx, sent in tqdm(enumerate(batch[0])):
            # print(sent)
            translation, layer_dict = self.model.translate(sent)
            # print(translation)
            # print(layer_dict.keys())
            # break
            self.make_dir_save_enc(layer_dict, idx, True)

    # ToDo
    def sanity_check(self, data_features_dir):
        filenames = natsorted(next(walk(self.path_save_encs + data_features_dir), (None, None, []))[2])
        features_l1 = np.load(self.path_save_encs + data_features_dir + filenames[0], allow_pickle=True)
        features_l2 = np.load(self.path_save_encs + data_features_dir + filenames[1], allow_pickle=True)
        features_l3 = np.load(self.path_save_encs + data_features_dir + filenames[2], allow_pickle=True)
        features_l4 = np.load(self.path_save_encs + data_features_dir + filenames[3], allow_pickle=True)
        features_l5 = np.load(self.path_save_encs + data_features_dir + filenames[4], allow_pickle=True)
        features_l6 = np.load(self.path_save_encs + data_features_dir + filenames[5], allow_pickle=True)

        print(features_l2[0][:10])
        print(features_l4[0][:10])

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


if __name__ == '__main__':
    tasks_dict = pd.read_csv('tasks_server.csv', index_col=0)
    # tasks_dict = pd.read_csv('tasks_lokal.csv', index_col=0)

    task_list = ['SUBJ', 'OBJ', 'TENSE', 'BIGRAM']
    # task_list = ['TENSE']

    # always spezify greatest value first; used to create encodings dataset
    data_size_list = [10000, 1000]
    # data_size_list = [200, 100]

    create_encodings = True
    # collect encodings from every layer, save every sentence in single file
    do_translation = True
    # read in all sentence encodings for layer n; get mean array for sentence tokens in layer n; save array
    do_avg_tensor = True
    # create scores for arrays
    classify_arrays = True
    # check if mean tensors are equal across layers
    sanity_check = False

    create_plots = False
    plot_avg_f_t = False
    plot_v_vs_t = True

    for task in task_list:
        path_in = tasks_dict[task]['path_in']
        path_out = tasks_dict[task]['path_out']

        if create_encodings:
            RunVisrep = VisRepEncodings(path_in, path_out)

            for m_type in ('t', 'v'):

                if m_type == 'v':
                    RunVisrep.make_vis_model(m_type)
                else:
                    RunVisrep.make_text_model(m_type)

                if sanity_check:
                    RunVisrep.sanity_check('results/')
                    break

                if do_translation:
                    RunVisrep.translate(RunVisrep.read_in_raw_data(data_size_list[0]))
                if do_avg_tensor:
                    RunVisrep.read_in_avg_enc_data(True)

                if classify_arrays:
                    for data_size in data_size_list:
                        results, dummy_results = RunVisrep.logistic_regression_classifier('results/', 'raw_labels.npy',
                                                                                          data_size)
                        results_f_t, dummy_results_f_t = RunVisrep.logistic_regression_classifier('results_f_t/',
                                                                                                  'raw_labels.npy',
                                                                                                  data_size)
                        results_all = {'avg': results, 'f_t': results_f_t, 'dummy': dummy_results}
                        df = pd.DataFrame.from_dict(results_all)
                        df.to_csv(path_out + m_type + '/' + m_type + '_' + task + '_' + str(data_size) + '.csv')

        if create_plots:
            print('\n Creating plots...\n')
            if plot_avg_f_t:
                for m_type in ('v', 't'):
                    for data_size in data_size_list:
                        df_in = pd.read_csv(path_out + m_type + '/' + m_type + '_' + task + '_' + str(data_size) +
                                            '.csv', index_col=0)
                        plot_results_avg_f_t(task, path_out, m_type, df_in, data_size)
            if plot_v_vs_t:
                for data_size in data_size_list:
                    df_v = pd.read_csv(path_out + 'v/v_' + task + '_' + str(data_size) + '.csv', index_col=0)
                    df_t = pd.read_csv(path_out + 't/t_' + task + '_' + str(data_size) + '.csv', index_col=0)

                    plot_results_v_vs_t(task, path_out, df_v, df_t, data_size)


