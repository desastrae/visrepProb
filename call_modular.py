from plot_models import *
from modular_run_tr import *

if __name__ == '__main__':
    # tasks_dict = pd.read_csv('tasks_server.csv', index_col=0)
    tasks_dict = pd.read_csv('tasks_lokal.csv', index_col=0)

    task_list = ['SUBJ', 'OBJ', 'TENSE', 'BIGRAM']
    # task_list = ['BIGRAM']

    # always spezify greatest value first; used to create encodings dataset
    # data_size_list = [10000, 1000]
    data_size_list = [20, 10]

    # create csv for majority class
    maj_class = False  # True

    # Start extraction process:
    # to obtain encodings for text and visual model; create avg np array; classify encodings for probing task.
    create_encodings = True

    # read in raw data into pd dataframe, write majority class to csv
    read_raw_data = False  # True

    # collect encodings from every layer, save every sentence in single file
    do_translation = False  # True
    # read in all sentence encodings for layer n; get mean array for sentence tokens in layer n; save array
    do_avg_tensor = False  # True
    # create scores for arrays
    classify_arrays = True
    # check if mean tensors are equal across layers
    sanity_check = False

    create_plots = False
    plot_avg_f_t = False
    plot_v_vs_t = False  # True

    if maj_class:
        maj_class_dict = defaultdict()

        for task in task_list:
            task_maj_class_dict = defaultdict()

            path_in = tasks_dict[task]['path_in']

            for data_size in data_size_list:
                df = pd.read_csv(path_in, delimiter='\t', header=None)[:data_size]
                val_df = '   '.join(str(df[1].value_counts()).split(',')[0].split('\n')[:2])
                task_maj_class_dict[str(data_size)] = val_df

            maj_class_dict[task] = task_maj_class_dict

        maj_class_df = pd.DataFrame.from_dict(maj_class_dict)

        maj_class_df.to_csv(tasks_dict['MAJ_CLASS']['path_out'] + '/majority_class.csv')

    for task in task_list:
        path_in_train = tasks_dict[task]['path_in_train']
        path_in_test = tasks_dict[task]['path_in_test']
        path_out = tasks_dict[task]['path_out']

        if create_encodings:
            RunVisrep = VisRepEncodings(path_in_train, path_out)

            if read_raw_data:
                raw_data_train = RunVisrep.read_in_raw_data(data_size_list[0], 0.75, 'train')
                raw_data_test = RunVisrep.read_in_raw_data(data_size_list[0], 0.25, 'test')

                for m_type in ('t', 'v'):

                    if m_type == 'v':
                        RunVisrep.make_vis_model(m_type)
                    else:
                        RunVisrep.make_text_model(m_type)

                    if do_translation:
                        RunVisrep.translate(raw_data_train, 'train')
                        RunVisrep.translate(raw_data_test, 'test')
                    if do_avg_tensor:
                        RunVisrep.read_in_avg_enc_data('train/')
                        RunVisrep.read_in_avg_enc_data('test/')

                    if sanity_check:
                        RunVisrep.sanity_check('results/')
                        break

            if classify_arrays:
                for m_type in ('t', 'v'):
                    for data_size in data_size_list:
                        print(data_size)
                        results, dummy_results = RunVisrep.logistic_regression_classifier(m_type + '/train/results/',
                                                                                          'train_raw_labels.npy',
                                                                                          m_type + '/test/results/',
                                                                                          'test_raw_labels.npy',
                                                                                          data_size)
                        # results_f_t, dummy_results_f_t = RunVisrep.logistic_regression_classifier('results_f_t/',
                        #                                                                           'raw_labels.npy',
                        #                                                                           data_size)
                        # results_all = {'avg': results, 'f_t': results_f_t, 'dummy': dummy_results}
                        results_all = {'avg': results, 'dummy': dummy_results}
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
                    maj_class_val = pd.read_csv(tasks_dict['MAJ_CLASS']['path_out'] + '/majority_class.csv', index_col=0)
                    # plot_results_v_vs_t(task, path_out, df_v, df_t, data_size, maj_class_val[task][data_size])
                    plot_results_v_vs_t(task, path_out, df_v, df_t, data_size)
