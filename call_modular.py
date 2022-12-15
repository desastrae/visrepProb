import pandas as pd
from plot_models import *
from modular_run_tr import *
import yaml
import sys
from collections import defaultdict
from configparser import ConfigParser
from subj_num import read_in_pickled_dict

if __name__ == '__main__':
    with open('config_visrep.yml') as config:
        config_dict = yaml.load(config, Loader=yaml.FullLoader)

        path_server_o_lokal = None

        if sys.argv[1] == 's':
            path_server_o_lokal = config_dict['server_path']
        elif sys.argv[1] == 'l':
            path_server_o_lokal = config_dict['lokal_path']
        else:
            print('Parameter ' + str(sys.argv[1]) + ' not existent.')
            exit(0)

        # TODO
        # path_tasks = None
        #
        # if config_dict['task'] == 'new_classifier':
        #     path_tasks = path_server_o_lokal + config_dict['xprobe_path_in']
        # elif config_dict['task'] == 'create_encs_and_test_noise':
        #     path_tasks = path_server_o_lokal + config_dict['noise_test_path_out']
        # elif config_dict['task'] == 'test_data':
        #     path_tasks = None

        task_list = config_dict['tasks']
        data_size_list = sorted(config_dict['dataset_size'], reverse=True)

        # create csv for majority class
        maj_class = False  # True

        # Start extraction process:
        # to obtain encodings for text and visual model; create avg np array; classify encodings for probing task.
        create_encodings = False
        create_encodings_test = True

        # read in raw data into pd dataframe, write majority class to csv
        read_raw_data = False  # True
        # collect encodings from every layer, save every sentence in single file
        do_translation = True
        # read in all sentence encodings for layer n; get mean array for sentence tokens in layer n; save array
        do_avg_tensor = True

        classify = False  # True
        # train classifier & create scores for arrays
        classify_arrays = False  # True
        # check if mean tensors are equal across layers
        sanity_check = False

        # Load saved model; classify test set
        saved_classifier = True

        # Create Plots
        create_plots = True
        plot_avg_f_t = False
        plot_v_vs_t = False
        plot_stack_plots = True

        # adapt / check if needed; NOT working
        if maj_class:
            maj_class_dict = defaultdict()

            for task in task_list:
                task_maj_class_dict = defaultdict()

                # path_in = tasks_dict[task]['path_in']
                path_in = path_server_o_lokal + config_dict['xprobe_path_in']

                for data_size in data_size_list:
                    df = pd.read_csv(path_in, delimiter='\t', header=None, encoding='utf-8')[:data_size]
                    val_df = '   '.join(str(df[1].value_counts()).split(',')[0].split('\n')[:2])
                    task_maj_class_dict[str(data_size)] = val_df

                maj_class_dict[task] = task_maj_class_dict

            # maj_class_df = pd.DataFrame.from_dict(maj_class_dict)

            # maj_class_df.to_csv(tasks_dict['MAJ_CLASS']['path_out'] + '/majority_class.csv')

        # main loop
        for task in task_list:

            if create_encodings:

                if read_raw_data:
                    path_in_train = path_server_o_lokal + config_dict['xprobe_path_in'] \
                                    + config_dict['xprobe_train_file']
                    path_in_test = path_server_o_lokal + config_dict['xprobe_path_in'] + config_dict['xprobe_test_file']
                    path_out = path_server_o_lokal + config_dict['data_path_in']

                    RunVisrep = VisRepEncodings(path_in_train, path_out, config_dict)

                    raw_data_train = RunVisrep.read_in_raw_data(data_size_list[0], 0.75, 'train', 'raw')
                    raw_data_test = RunVisrep.read_in_raw_data(data_size_list[0], 0.25, 'test', 'raw')

                    for m_type in ('t', 'v'):

                        if m_type == 'v':
                            RunVisrep.make_vis_model(m_type)
                        else:
                            RunVisrep.make_text_model(m_type)

                        if do_translation:
                            RunVisrep.translate_save(raw_data_train, 'train', True)
                            RunVisrep.translate_save(raw_data_test, 'test', True)
                        if do_avg_tensor:
                            RunVisrep.read_in_avg_enc_data('train/', 'clean')
                            RunVisrep.read_in_avg_enc_data('test/', 'clean')

                        if sanity_check:
                            RunVisrep.sanity_check('results/')
                            break

                if create_encodings_test:
                    path_in_test = path_server_o_lokal + config_dict['data_path_in'] + task + '/'
                    path_out = path_server_o_lokal + config_dict['noise_test_path_out'] + task + '/'

                    RunVisrep = VisRepEncodings(config_dict, path_in_test, path_out)

                    print('noise_data_test', path_server_o_lokal + config_dict['data_path_in'] + task + '/'
                                                  + config_dict['noise_test_file_out'])
                    noise_data_test = pd.read_csv(path_server_o_lokal + config_dict['data_path_in'] + task + '/'
                                                  + config_dict['noise_test_file_out'], delimiter=';')

                    print(noise_data_test.columns)

                    for col in noise_data_test.columns:
                        print(noise_data_test[col])
                        print(col)

                        for m_type in ('t', 'v'):

                            if m_type == 'v':
                                RunVisrep.make_vis_model(m_type)
                            else:
                                RunVisrep.make_text_model(m_type)

                            if do_translation:
                                RunVisrep.translate_save(noise_data_test[col], 'test', col)
                            if do_avg_tensor:
                                RunVisrep.read_in_avg_enc_data('test/', col)

            if classify:

                if classify_arrays:
                    print('Training Classifier & Evaluating Data...\n')
                    for m_type in ('t', 'v'):
                        for data_size in data_size_list:
                            print(data_size)
                            results, dummy_results = RunVisrep.logistic_regression_classifier(m_type, m_type +
                                                                                              '/train/results/',
                                                                                              'train_raw_labels.npy',
                                                                                              m_type + '/test/results/',
                                                                                              'test_raw_labels.npy',
                                                                                              data_size)
                            results_all = {'avg': results, 'dummy': dummy_results}
                            df = pd.DataFrame.from_dict(results_all)
                            df.to_csv(path_out + m_type + '/' + m_type + '_' + task + '_' + str(data_size) + '.csv')

                if saved_classifier:
                    data_size_list = [10000]  # , 1000]
                    path_in_test = path_server_o_lokal + config_dict['data_path_in'] + task + '/'
                    path_out = path_server_o_lokal + config_dict['noise_test_path_out'] + task + '/'

                    RunVisrep = VisRepEncodings(config_dict, path_in_test, path_out)

                    print('Loading saved Classifier & Evaluating Data...\n')
                    for m_type in ('t', 'v'):
                        folder_name_path = path_server_o_lokal + config_dict['noise_test_path_out'] + task \
                                           + '/' + m_type + '/test/results/'
                        noise_folder_names = natsorted(next(walk(folder_name_path), (None, [], None))[1])
                        results_all = defaultdict()
                        for data_size in data_size_list:
                            for noise_folder in noise_folder_names:
                                path_avg_encs = path_server_o_lokal + config_dict['noise_test_path_out'] + task + \
                                                '/' + m_type + '/test/results/' + noise_folder + '/'
                                path_classifier = path_server_o_lokal + config_dict['data_path_in'] + task + '/' \
                                                  + m_type + '/' + config_dict['path_saved_classifier']
                                path_labels = path_server_o_lokal + config_dict['data_path_in'] + task + '/' \
                                                  + config_dict['noise_test_labels_in']
                                results = RunVisrep.load_classifier_model_load_avg_encs(path_avg_encs, path_classifier,
                                                                                        path_labels)
                                results_all[noise_folder] = results
                        df = pd.DataFrame.from_dict(results_all)
                        noise_info_str = '_'.join(config_dict['noise_type']) + '_' + '_'.join(config_dict['noise_perc'])
                        df.to_csv(path_out + 'noise_' + noise_info_str + m_type + '_' + task + '_' + str(data_size)
                                  + '.csv')

            if create_plots:
                # path_out = path_server_o_lokal + config_dict['data_path_in']
                path_out = path_server_o_lokal + config_dict['noise_test_path_out'] + task + '/'
                # filenames = natsorted(next(walk(path_out), (None, None, []))[2])
                get_filenames = next(walk(path_out), (None, None, []))[2]
                filenames = list( filter(lambda k: '.csv' in k, get_filenames))
                print('filenames', filenames)
                print('\n Creating plots...\n')
                if plot_avg_f_t:
                    for file in filenames:
                        df_in = pd.read_csv(path_out + file, index_col=0)
                        plot_results_avg_f_t(task, path_out, m_type, df_in, data_size)
                if plot_v_vs_t:
                    for file in filenames:
                        df_v = pd.read_csv(path_out + file, index_col=0)
                        df_t = pd.read_csv(path_out + file, index_col=0)
                        # maj_class_val = pd.read_csv(tasks_dict['MAJ_CLASS']['path_out'] + '/majority_class.csv', index_col=0)
                        # plot_results_v_vs_t(task, path_out, df_v, df_t, data_size, maj_class_val[task][data_size])
                        # results_name = path_out + config_dict['train_test'][0] + '/results/' + data_name + '/'
                        # layers_list = listdir(folder_name)

                        # plot_results_v_vs_t(task, path_out, df_v, df_t, data_size)

                if plot_stack_plots:
                    for file in filenames:
                        print('\n Creating plots...\n')
                        path_old_scores = path_server_o_lokal + config_dict['data_path_in'] + task
                        file_v_old_scores = path_old_scores + '/v/v_' + task + '_' + str(10000) + '.csv'
                        file_t_old_scores = path_old_scores + '/t/t_' + task + '_' + str(10000) + '.csv'

                        # TODO: adpat file names dynamically !
                        df_v_new = pd.read_csv(path_out + 'noise_v_' + task + '_' + str(10000) + '.csv', index_col=0)
                        df_t_new = pd.read_csv(path_out + 'noise_t_' + task + '_' + str(10000) + '.csv', index_col=0)

                        df_v_old = pd.read_csv(file_v_old_scores, index_col=0)
                        df_t_old = pd.read_csv(file_t_old_scores, index_col=0)

                        train_subj_obj_dict = read_in_pickled_dict(
                            path_out + 'eval_' + task + '_train_train_raw_sentences.npy')
                        train_subj_obj_dict_sum = sum(train_subj_obj_dict.values())
                        test_subj_obj_dict = read_in_pickled_dict(
                            path_out + 'eval_' + task + '_test_test_raw_sentences.npy')
                        test_subj_obj_dict_sum = sum(test_subj_obj_dict.values())

                        df_header = list(df_t_new.columns)
                        print('df_header', df_header)

                        if config_dict['config'] == 'subj_obj_data':
                            df_header_adapted = [name.split('3')[0] + '-train_'
                                                 + str(
                                "{:.2f}".format((train_subj_obj_dict[name.split('3')[0].split('BJ_')[1]] /
                                                 train_subj_obj_dict_sum) * 100.0)) + '%'
                                                 + '-test_'
                                                 + str(
                                "{:.2f}".format((test_subj_obj_dict[name.split('3')[0].split('BJ_')[1]] /
                                                 test_subj_obj_dict_sum) * 100.0)) + '%'
                                                 for name in df_header]
                            # print('df_header_adapted', df_header_adapted)

                            for df_n in [df_t_new, df_v_new]:
                                # print('df_n.columns', df_n.columns, '\n', df_header_adapted)
                                for name in df_header_adapted:
                                    df_n.rename({str(name.split('-')[0] + '3.txt'): name}, errors="raise", axis=1,
                                                inplace=True)

                        if config_dict['config'] == 'noise':
                            for noise_type in config_dict['noise_type']:
                                df_t_filter = df_t_new.filter(regex=noise_type)
                                print(df_t_filter)
                                df_v_filter = df_v_new.filter(regex=noise_type)
                                path_out_noise = path_out + noise_type + '_'
                                if len(list(df_v_filter.columns)) > 1:
                                    stack_plots(task, path_out_noise, df_v_filter, df_t_filter, df_v_old, df_t_old,
                                                config_dict)
                                elif len(list(df_v_filter.columns)) == 1:
                                    one_plot_old_new(task, path_out_noise, df_v_filter, df_t_filter, df_v_old, df_t_old,
                                                     config_dict)
                        else:
                            if len(list(df_v_new.columns)) > 1:
                                stack_plots(task, path_out, df_v_new, df_t_new, df_v_old, df_t_old, config_dict)
                            elif len(list(df_v_filter.columns)) == 1:
                                one_plot_old_new(task, path_out, df_v_new, df_t_new, df_v_old, df_t_old, config_dict)
