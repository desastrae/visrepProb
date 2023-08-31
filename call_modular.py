import pandas as pd
from plot_models import *
from modular_run_tr import *
import yaml
import sys
from collections import defaultdict
from configparser import ConfigParser
from subj_num import read_in_pickled_dict
from sem_xml_data import get_train_test_sem, get_sem_data_dirs


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

        # path_server_o_lokal = config_dict['lokal_path']

        # TODO
        # path_tasks = None
        #
        # if config_dict['task'] == 'new_classifier':
        #     path_tasks = path_server_o_lokal + config_dict['xprobe_path_in']
        # elif config_dict['task'] == 'create_encs_and_test_noise':
        #     path_tasks = path_server_o_lokal + config_dict['noise_test_path_out']
        # elif config_dict['task'] == 'test_data':
        #     path_tasks = None

        # task_list = config_dict['tasks']
        task_list = config_dict['tasks_word']
        data_size_list = sorted(config_dict['dataset_size'], reverse=True)

        # create csv for majority class
        maj_class = False  # True

        # Start extraction process:
        # to obtain encodings for text and visual model; create avg np array; classify encodings for probing task.
        create_encodings = False  # True
        create_encodings_test = True  # False

        # read in raw data into pd dataframe, write majority class to csv
        read_raw_data = False
        # collect encodings from every layer, save every sentence in single file
        do_translation = True  # False
        # SENT: read in all sentence encodings for layer n; get mean array for sentence tokens in layer n; save array
        # OR
        # WORD: read in all sentence encodings for layer n; get mean array for word in sentence tokens in layer n;
        # save word-level arrays as matrix; each row is a sentence containing word-level encodings
        do_avg_tensor = True

        classify = True
        # train classifier & create scores for arrays
        classify_arrays = False  # True
        # test results with normalized embeddings
        classify_norm = False  # True
        # check if mean tensors are equal across layers
        sanity_check = False
        # Load saved model; classify test set
        saved_classifier = True

        # Create Plots
        create_plots = False  # True
        plot_avg_f_t = False
        plot_v_vs_t = False
        plot_prob_tasks = False  # True
        plot_stack_plots = True  # False
        plot_per_layer = False  # True

        # TODO: adapt / check if needed; NOT working
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

                # TODO: change paths
                if config_dict['sent_word_prob'] == 'sent' and task != 'pos':
                    path_in_train = path_server_o_lokal + config_dict['xprobe_path_in'] \
                                    + config_dict['xprobe_train_file']
                    path_in_test = path_server_o_lokal + config_dict['xprobe_path_in'] + config_dict['xprobe_test_file']
                    path_out = path_server_o_lokal + config_dict['data_path_in']

                    RunVisrep = VisRepEncodings(config_dict, path_in_train, path_out, task)

                elif config_dict['sent_word_prob'] == 'word':
                    if task == 'sem':
                        path_in_file = path_server_o_lokal + config_dict['data_path_in'] + config_dict['sem_path_in']
                    elif task == 'dep':
                        path_in_file = path_server_o_lokal + config_dict['data_path_in'] + \
                                       config_dict['UD_path_in'] + config_dict['UD_file']
                    path_out = path_server_o_lokal + config_dict['data_path_in'] + task + '/'

                    try:
                        os.mkdir(path_out)
                    except OSError as error:
                        # print(error)
                        pass

                    RunVisrep = VisRepEncodings(config_dict, path_in_file, path_out, task)

                # read in original dataformat, read in & preprocess for translation pipeline...
                if read_raw_data and not create_encodings_test:

                    if config_dict['sent_word_prob'] == 'sent':
                        raw_data_train = RunVisrep.read_in_raw_data(data_size_list[0], 0.75, 'train', 'raw')
                        raw_data_test = RunVisrep.read_in_raw_data(data_size_list[0], 0.25, 'test', 'raw')

                    elif config_dict['sent_word_prob'] == 'word' and task == 'dep':
                        # raw_sent_pos_data = RunVisrep.read_pos_raw_data(path_in_file)
                        # path_in_file = path_server_o_lokal + config_dict['data_path_in'] + config_dict['UD_path_in'] \
                        #                + config_dict['UD_file']
                        raw_sent_pos_data = RunVisrep.read_UD_data(path_in_file)
                        # print('raw_sent_pos_data', len(raw_sent_pos_data), 'data_size_list[0] * 0.75',
                        # data_size_list[0] * 0.75)
                        # sents_list, pos_list = [list(zip(*sent)) for sent in raw_sent_pos_data[0]]

                        # CHAR ½ verursachte Fehler: Sätze mit diesem Character werden entfernt

                        raw_data_train = raw_sent_pos_data[0:int(data_size_list[0] * 0.75)]
                        raw_data_test = raw_sent_pos_data[int(data_size_list[0] * 0.75):]

                    elif config_dict['sent_word_prob'] == 'word' and task == 'sem':
                        # gold, silver, bronze = get_sem_data_dirs(path_in_file)
                        # raw_data_train, raw_data_test = get_train_test_sem(path_in_file, gold, silver, bronze)

                        gold = get_sem_data_dirs(path_in_file)
                        raw_sem_data = get_train_test_sem(path_in_file, gold)
                        raw_data_train = raw_sem_data[0:int(len(raw_sem_data) * 0.75)]
                        raw_data_test = raw_sem_data[int(len(raw_sem_data) * 0.75):]

                for m_type in ('v', 't'):

                    if m_type == 'v':
                        RunVisrep.make_vis_model(m_type)
                    else:
                        RunVisrep.make_text_model(m_type)

                    # print(do_avg_tensor, config_dict['sent_word_prob'])

                    if do_translation and config_dict['sent_word_prob'] == 'sent' and not create_encodings_test:
                        print('Translate sentences at sentence-level...')
                        RunVisrep.translate_save(raw_data_train, 'train', task)
                        RunVisrep.translate_save(raw_data_test, 'test', task)
                    if do_translation and config_dict['sent_word_prob'] == 'word' and not create_encodings_test:
                        print('Translate sentences at word-level...')
                        RunVisrep.translate_word_level_save(raw_data_train, 'train', task)
                        RunVisrep.translate_word_level_save(raw_data_test, 'test', task)
                    if do_avg_tensor and config_dict['sent_word_prob'] == 'sent' and not create_encodings_test:
                        print('Create averaged encodings at sentence-level...\n')
                        RunVisrep.read_in_avg_enc_data('train/', 'clean')
                        RunVisrep.read_in_avg_enc_data('test/', 'clean')
                    if do_avg_tensor and config_dict['sent_word_prob'] == 'word' and not create_encodings_test:
                        print('Create averaged encodings at word-level...\n')
                        RunVisrep.read_in_word_level_make_matrix('train')
                        RunVisrep.read_in_word_level_make_matrix('test')
                    if sanity_check:
                        RunVisrep.sanity_check('results/')
                        break

                    if create_encodings_test and config_dict['sent_word_prob'] == 'sent':
                        path_in_test = path_server_o_lokal + config_dict['data_path_in'] + task + '/'
                        path_out = path_server_o_lokal + config_dict['noise_test_path_out'] + task + '/'

                        RunVisrep = VisRepEncodings(config_dict, path_in_test, path_out, task)

                        print('noise_data_test', path_server_o_lokal + config_dict['data_path_in'] + task + '/'
                                                      + config_dict['noise_test_file_out'])
                        noise_data_test = pd.read_csv(path_server_o_lokal + config_dict['data_path_in'] + task + '/'
                                                      + config_dict['noise_test_file_out'], delimiter=';')

                        print(noise_data_test.columns)

                        for col in noise_data_test.columns:
                            print(noise_data_test[col])
                            print(col)

                            for m_type in ('v', 't'):

                                if m_type == 'v':
                                    RunVisrep.make_vis_model(m_type)
                                else:
                                    RunVisrep.make_text_model(m_type)

                                if do_translation:
                                    RunVisrep.translate_save(noise_data_test[col], 'test', col)
                                if do_avg_tensor:
                                    RunVisrep.read_in_avg_enc_data('test/', col)

                    if create_encodings_test and config_dict['sent_word_prob'] == 'word':
                        if task == 'dep':
                            path_in_test = path_server_o_lokal + config_dict['data_path_in'] + config_dict['UD_path_in'] \
                                           + 'noise_data/'
                            path_out = path_server_o_lokal + config_dict['data_path_in'] + task + '/' # + m_type # + \
                                       # '/test/'
                        elif task == 'sem':
                            path_in_test = path_server_o_lokal + config_dict['data_path_in'] + config_dict['sem_path_in'] \
                                           + 'noise_data/'
                            path_out = path_server_o_lokal + config_dict['data_path_in'] + task + '/' # + m_type # + \
                                       # '/test/'

                        noise_filenames = natsorted(next(walk(path_in_test), (None, None, []))[2])

                        for noise_type in config_dict['noise_type']:
                            print('Creating noise files for noise type ' + noise_type + '...\n')
                            RunVisrep = VisRepEncodings(config_dict, path_in_test, path_out, task)
                            if m_type == 'v':
                                RunVisrep.make_vis_model(m_type)
                            else:
                                RunVisrep.make_text_model(m_type)
                            noise_type_files = sorted(filter(lambda file: noise_type in file, noise_filenames))

                            for file in tqdm(noise_type_files):
                                file_name = file.split('.t')
                                print(file_name[0])
                                with open(path_in_test + file) as noise_file:
                                    file_data = noise_file.read().splitlines()
                                    if do_translation:
                                        RunVisrep.translate_save_noise(file_data, 'test', file_name[0])
                                    if do_avg_tensor:
                                        RunVisrep.read_in_word_level_make_matrix('test', file_name[0])
                                    # if create_encodings_test and do_avg_tensor:
                                    #     RunVisrep.read_in_word_level_make_matrix('test')

            if classify:
                if config_dict['sent_word_prob'] == 'sent':
                    path_in_train = path_server_o_lokal + config_dict['xprobe_path_in'] \
                                    + config_dict['xprobe_train_file']
                    path_in_test = path_server_o_lokal + config_dict['xprobe_path_in'] + config_dict['xprobe_test_file']
                    path_out = path_server_o_lokal + config_dict['data_path_in'] + task + '/'

                    RunVisrep = VisRepEncodings(config_dict, path_in_train, path_out, task)

                elif config_dict['sent_word_prob'] == 'word':
                    path_in_file = path_server_o_lokal + config_dict['data_path_in'] + \
                                   config_dict['UD_path_in'] + config_dict['UD_file']
                    path_out = path_server_o_lokal + config_dict['data_path_in'] + task + '/'

                for m_type in ('v', 't'):
                    print('MODEL:', m_type, '\n\n')
                    RunVisrep = VisRepEncodings(config_dict, path_in_file, path_out + m_type + '/', task)

                    if m_type == 'v':
                        RunVisrep.make_vis_model(m_type)
                    else:
                        RunVisrep.make_text_model(m_type)

                    if classify_arrays:
                        print('Training Classifier & Evaluating Data...\n')
                        if config_dict['classifier'] == 'mlp' and not classify_norm:
                            results = RunVisrep.mlp_classifier(m_type, data_size_list[0])
                        if config_dict['classifier'] == 'mlp' and classify_norm:
                            print('classify norm mlp...')
                            results = RunVisrep.mlp_classifier(m_type, data_size_list[0], True)
                        elif config_dict['classifier'] == 'lr' and not classify_norm:
                            results = RunVisrep.log_reg_no_dict_classifier(m_type, data_size_list[0])
                        elif config_dict['classifier'] == 'lr' and classify_norm:
                            print('classify norm lr...')
                            results = RunVisrep.log_reg_no_dict_classifier(m_type, data_size_list[0], True)
                        else:
                            print('Unknown classifier...')
                            sys.exit()
                        # results_all = {'avg': results, 'dummy': dummy_results}
                        df = pd.DataFrame.from_dict(results)
                        df.to_csv(path_out + m_type + '/' + config_dict['classifier'] + '_' +
                                  config_dict['sent_word_prob'] + '_' + m_type + '_' + task + '_' +
                                  str(data_size_list[0]) + '.csv')

                    if saved_classifier:
                        # data_size_list = [10000]  # , 1000]
                        data_size_list = config_dict['dataset_size'][:1]
                        print('data_size_list: ', str(data_size_list[0]))
                        print(path_out + 'f1_noise_' + m_type + '_' + task + '_' + str(data_size_list[0]) + '.csv')
                        path_in_test = path_server_o_lokal + config_dict['data_path_in'] + task + '/'
                        path_out = path_server_o_lokal + config_dict['data_path_in'] + task + '/' + m_type + '/'

                        RunVisrep = VisRepEncodings(config_dict, path_in_test, path_out, task)

                        print('Loading saved Classifier & Evaluating Data...\n')
                        # for m_type in ('v', 't')[:1]:
                        #     folder_name_path = path_server_o_lokal + config_dict['noise_test_path_out'] + task \
                        #                        + '/' + m_type + '/test/results/'
                        results_all = defaultdict()
                        results_all_f1 = defaultdict()
                        if config_dict['config'] == 'noise':
                            path_noise = path_out + 'test/results/noise/'
                            # noise_folder_names = natsorted(next(walk(folder_name_path), (None, [], None))[1])
                            noise_folder_names = natsorted(next(walk(path_noise),
                                                                (None, [], None))[1])
                            noise_info_str = '_'.join(config_dict['noise_type'])
                            noise_dict = defaultdict()
                            noise_f1_dict = defaultdict()

                            if task == 'dep':
                                for dep_task in ['xpos', 'upos', 'dep']:
                                    print(dep_task)
                                    for noise_folder in noise_folder_names:
                                        path_labels = path_out + 'test_' + dep_task + '_all_labels_array.npy'
                                        path_avg_encs = path_noise + noise_folder + '/'
                                        path_classifier = path_out + str(config_dict['classifier']) + '_sav/'
                                        results, results_f1 = RunVisrep.load_classifier_model_word_level(dep_task,
                                                                                                         path_avg_encs,
                                                                                                         path_classifier,
                                                                                                         path_labels)
                                        noise_dict[noise_folder] = results
                                        print(noise_dict)
                                        noise_f1_dict[noise_folder] = results_f1
                                    pd.DataFrame.from_dict(noise_dict).to_csv(
                                        path_out + dep_task + '_noise_' + noise_info_str + '_' + m_type + '_' +
                                        task + '_' + str(data_size_list[0]) + '.csv')
                                    pd.DataFrame.from_dict(noise_f1_dict).to_csv(
                                        path_out + dep_task + '_f1_noise_' + noise_info_str + '_' + m_type + '_' +
                                        task + '_' + str(data_size_list[0]) + '.csv')

                            elif task == 'sem':
                                for noise_folder in noise_folder_names:
                                    path_labels = path_out + 'test_sem_all_labels_array.npy'
                                    path_avg_encs = path_noise + noise_folder + '/'
                                    path_classifier = path_out + str(config_dict['classifier']) + '_sav/'
                                    results, results_f1 = RunVisrep.load_classifier_model_word_level(task,
                                                                                                     path_avg_encs,
                                                                                                     path_classifier,
                                                                                                     path_labels)
                                    noise_dict[noise_folder] = results
                                    noise_f1_dict[noise_folder] = results_f1
                                pd.DataFrame(noise_dict).to_csv(
                                    path_out + 'sem_noise_' + noise_info_str + '_' + m_type + '_' +
                                    task + '_' + str(data_size_list[0]) + '.csv')
                                pd.DataFrame(noise_f1_dict).to_csv(
                                    path_out + 'sem_f1_noise_' + noise_info_str + '_' + m_type + '_' +
                                    task + '_' + str(data_size_list[0]) + '.csv')

                        else:
                            path_avg_encs = path_out + 'test/results/clean/'
                            path_classifier = path_out + config_dict['classifier'] + '_sav/'

                            task_dict = defaultdict()
                            task_dict_f1 = defaultdict()
                            if task == 'dep':
                                for dep_task in ['xpos', 'upos', 'dep']:
                                    print(dep_task)
                                    path_labels = path_out + 'test_' + dep_task + '_all_labels_array.npy'
                                    path_out_class_report = path_out + config_dict['classifier'] + \
                                                            '_classification_report_' + dep_task + '.txt'
                                    results, results_f1 = RunVisrep.load_classifier_model_word_level(dep_task,
                                                                                         path_avg_encs,
                                                                                         path_classifier, path_labels)
                                                                                         # path_out_class_report)
                                    task_dict[dep_task] = results
                                    task_dict_f1[dep_task] = results_f1
                                    # results_all[noise_folder] = results
                                info_str = config_dict['classifier'] + '_' + m_type + '_dep'
                                info_str_f1 = config_dict['classifier'] + '_' + m_type + '_dep_f1-scores'
                                print(task_dict)
                                pd.DataFrame(task_dict).to_csv(path_out + info_str + '.csv')
                                pd.DataFrame(task_dict_f1).to_csv(path_out + info_str + '.csv')
                            elif task == 'sem':
                                path_labels = path_out + 'test_sem_all_labels_array.npy'
                                path_out_class_report = path_out + config_dict['classifier'] + \
                                                        '_classification_report_sem.txt'
                                results, results_f1 = RunVisrep.load_classifier_model_word_level(task, path_avg_encs,
                                                                                     path_classifier, path_labels)
                                                                                     # path_out_class_report)
                                # results_all[noise_folder] = results
                                task_dict[task] = results
                                task_dict_f1[task] = results_f1
                                info_str = config_dict['classifier'] + '_' + m_type + '_sem'
                                info_str_f1 = config_dict['classifier'] + '_' + m_type + '_sem_f1-scores'
                                pd.DataFrame(task_dict).to_csv(path_out + info_str + '.csv')
                                pd.DataFrame(task_dict_f1).to_csv(path_out + info_str_f1 + '.csv')

            if create_plots:
                # path_out = path_server_o_lokal + config_dict['data_path_in'] + task + '/'
                path_out = path_server_o_lokal + config_dict['data_path_in']
                # filenames = natsorted(next(walk(path_out), (None, None, []))[2])
                get_filenames = next(walk(path_out), (None, None, []))[2]
                all_filenames = list(filter(lambda k: '.csv' in k, get_filenames))
                f1_filenames = list(filter(lambda g: 'f1' in g, all_filenames))
                filenames = list(filter(lambda g: 'norm' not in g, f1_filenames))
                norm_filenames = list(filter(lambda g: 'norm' in g, f1_filenames))
                print('filenames', filenames)

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

                if plot_prob_tasks:

                    if config_dict['sent_word_prob'] == 'sent':
                        path_old_scores = path_server_o_lokal + config_dict['data_path_in']
                        file_v_scores = path_old_scores + 'v_prob_tasks_results.csv'
                        file_t_scores = path_old_scores + 't_prob_tasks_results.csv'
                    elif config_dict['sent_word_prob'] == 'word':  # and config_dict['tasks_word'][0] == 'dep':
                        # file_v_scores = path_out + config_dict['classifier'] + '_word_v_dep_10000.csv'
                        # file_t_scores = path_out + config_dict['classifier'] + '_word_t_dep_10000.csv'
                        cl_m = config_dict['classifier']
                        file_v_scores = path_out + list(filter(lambda mod: '_v_' in mod, (list(filter(lambda cl:
                                                                                                      cl_m in cl,
                                                                                           norm_filenames)))))[0]
                        file_t_scores = path_out + list(filter(lambda mod: '_t_' in mod, (list(filter(lambda cl:
                                                                                                      cl_m in cl,
                                                                                           norm_filenames)))))[0]
                        # print('filtered list', list(filter(lambda file: 'f1' in file, file_v_scores)))
                    df_v = pd.read_csv(file_v_scores, index_col=0, sep=',')
                    df_t = pd.read_csv(file_t_scores, index_col=0, sep=',')

                    print(df_v)

                    line_plot_prob_tasks_v1(df_v, df_t, config_dict['classifier'], path_out + config_dict['classifier']
                                            + '_norm_f1-scores_')
                    line_plot_prob_tasks_v2(df_v, df_t, config_dict['classifier'], path_out + config_dict['classifier']
                                            + '_norm_f1-scores_')

                # To-Do: Check this step!
                train_train_what = False

                # relevant for noise
                if plot_stack_plots:
                    if config_dict['sent_word_prob'] == 'sent':
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

                            # print('df_v_old', df_v_old)
                            # print('df_t_old', df_t_old)
                    elif config_dict['sent_word_prob'] == 'word':
                        file_v_scores = path_out + list(filter(lambda k: '_v_' in k, list(filter(lambda g: '10000' in g,
                                                                                                 all_filenames))))[0]
                        file_t_scores = path_out + list(filter(lambda k: '_t_' in k, list(filter(lambda g: '10000' in g,
                                                                                                 all_filenames))))[0]
                        df_v_no_noise = pd.read_csv(file_v_scores, index_col=0)
                        df_t_no_noise = pd.read_csv(file_t_scores, index_col=0)

                    if train_train_what:
                        train_subj_obj_dict = read_in_pickled_dict(
                            path_out + 'eval_' + task + '_train_train_raw_sentences.npy')
                        train_subj_obj_dict_sum = sum(train_subj_obj_dict.values())
                        test_subj_obj_dict = read_in_pickled_dict(
                            path_out + 'eval_' + task + '_test_test_raw_sentences.npy')
                        test_subj_obj_dict_sum = sum(test_subj_obj_dict.values())

                    df_header = list(df_t_no_noise.columns)
                    # print('df_header', df_header)

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

                    if config_dict['config'] == 'OLDnoise':
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
                    if config_dict['config'] == 'noise' and config_dict['sent_word_prob'] == 'sent':
                        for noise_type in config_dict['noise_type']:
                            df_t_filter = df_t_new.filter(regex=noise_type)
                            print(df_t_filter)
                            df_v_filter = df_v_new.filter(regex=noise_type)
                            path_out_noise = path_out + noise_type + '_'
                            line_plot_per_layer(task, path_out_noise, df_v_filter, df_t_filter, df_v_old, df_t_old,
                                                config_dict)

                    if config_dict['config'] == 'noise' and config_dict['sent_word_prob'] == 'word':
                        file_v_noise = path_out + list(filter(lambda k: '_v_' in k, list(filter(lambda g: '10000' in g,
                                                                                                 all_filenames))))[0]
                        file_t_noise = path_out + list(filter(lambda k: '_t_' in k, list(filter(lambda g: '10000' in g,
                                                                                                 all_filenames))))[0]
                        df_v_noise = pd.read_csv(file_v_noise, index_col=0)
                        df_t_noise = pd.read_csv(file_t_noise, index_col=0)
                        print('df_v_noise: ', df_v_noise)

                        for noise_type in config_dict['noise_type']:
                            df_t_filter = df_t_noise.filter(regex=noise_type)
                            print(df_t_filter)
                            df_v_filter = df_v_noise.filter(regex=noise_type)
                            path_out_noise = path_out + noise_type + '_'
                            line_plot_per_layer(task, path_out_noise, df_v_filter, df_t_filter, df_v_no_noise,
                                                df_t_no_noise, config_dict)
                    if config_dict['config'] == 'bleu':
                        df_bleu_noise = pd.read_csv(path_server_o_lokal + config_dict['path_file_bleu_scores'])
                        df_bleu_noise_mttt = pd.read_csv(
                            path_server_o_lokal + config_dict['path_file_bleu_scores_mttt'])

                        for noise_type in config_dict['noise_type']:
                            df_t_filter = df_t_new.filter(regex=noise_type)
                            # print(df_t_filter)
                            df_v_filter = df_v_new.filter(regex=noise_type)
                            df_bleu_filter = df_bleu_noise.filter(regex=noise_type)
                            df_bleu_filter_mttt = df_bleu_noise_mttt.filter(regex=noise_type)
                            print('df bleu \n', df_bleu_filter)
                            path_out_noise = path_out + noise_type + '_'
                            bleu_line_plot_per_layer(task, path_out_noise, df_v_filter, df_t_filter, config_dict,
                                                     df_bleu_filter, df_bleu_filter_mttt, noise_type, df_v_old['avg'],
                                                     df_t_old['avg'])

                    if config_dict['config'] == 'bleu_mttt':
                        df_bleu_noise_mttt = pd.read_csv(path_server_o_lokal + config_dict['path_file_bleu_scores_mttt'])

                        noise_types_list = config_dict['noise_type']
                        bleu_mttt_line_plot_per_layer(path_out, config_dict, df_bleu_noise_mttt, noise_types_list)

                    # if 'TODO':
                    #     if len(list(df_v_new.columns)) > 1:
                    #         stack_plots(task, path_out, df_v_new, df_t_new, df_v_old, df_t_old, config_dict)
                    #     elif len(list(df_v_filter.columns)) == 1:
                    #         one_plot_old_new(task, path_out, df_v_new, df_t_new, df_v_old, df_t_old, config_dict)
