from plot_models import *
from modular_run_tr import *
import os
from subj_num import read_in_pickled_dict
import yaml
import seaborn as sns


def line_plot_prob_tasks_v1(file_v, file_v_new, file_t, file_t_new, path_save):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 9.6))

    # Set the axes title font size
    plt.rc('axes', titlesize=22)  # Set the axes labels font size
    # Set the font size of the figure title
    plt.rc('figure', titlesize=25)

    x_v_axis = file_v.index.to_list()
    x_t_axis = file_t.index.to_list()

    colors_list = sns.color_palette('Paired')

    # colors_list = ["palegreen", "hotpink", "cornflowerblue", "golden"]
    colors_list_f = ['#f9d030', '#b8ee30', '#26dfd0', '#f62aa0']
    colors_list = sns.color_palette('colorblind', 8)
    # colors_list = ["royalblue", "darkorange", "green", "red", "gold", "turquoise", "magenta"]
    dot_symbols_list = ['o', 'v', '*', 'H', '^', 'x', 'D']

    prob_tasks_list = sorted(file_v_new.keys())

    ax1.plot(x_v_axis, file_v['avg'], linestyle='-', linewidth=7.0, color=colors_list[0], marker='o', markersize=15)
    ax2.plot(x_t_axis, file_t['avg'], linestyle='-', linewidth=7.0, color=colors_list[1], marker='v', markersize=15)

    # for task, color, symbol in zip(prob_tasks_list, colors_list, dot_symbols_list):
    for i in range(len(prob_tasks_list)):
        # print('layer ', layer[-1], '\n color ', color, '\n symbol ', symbol)
        # print(i*2, i*2-1, i*3-1)

        ax1.plot(x_v_axis, file_v_new[prob_tasks_list[i]], linestyle='-', color=colors_list_f[i], marker=dot_symbols_list[i],
                 markersize=12)
        ax2.plot(x_t_axis, file_t_new[prob_tasks_list[i]], linestyle='-', color=colors_list_f[i], marker=dot_symbols_list[i],
                 markersize=12)

    prob_tasks_list.insert(0, 'Original Test Results')

    # ax1.legend(prob_tasks_list, fontsize="10")
    # ax2.legend(prob_tasks_list, fontsize="10")

    ax1.set_xlabel('Layers', fontsize=18)
    ax2.set_xlabel('Layers', fontsize=18)

    ax1.set_ylabel('Probing Accuracy', fontsize=18)
    ax2.set_ylabel('Probing Accuracy', fontsize=18)

    ax1.set_title('Visual Encoder')
    ax2.set_title('Text Encoder')

    if file_v_new.columns[0].split('_')[0] == 'OBJ':
        task = 'obj_number'
        ax1.set_ylim([0.2, 1.05])
        ax2.set_ylim([0.2, 1.05])

        ax1.set_yticks(np.arange(0.2, 1.05, step=0.05))
        ax2.set_yticks(np.arange(0.2, 1.05, step=0.05))
    elif file_v_new.columns[0].split('_')[0] == 'SUBJ':
        task = 'subj_number'
        ax1.set_ylim([0.35, 1.05])
        ax2.set_ylim([0.35, 1.05])

        ax1.set_yticks(np.arange(0.35, 1.05, step=0.05))
        ax2.set_yticks(np.arange(0.35, 1.05, step=0.05))

    fig.suptitle('Data Reliability\nTask: ' + task + '\nLinear Regression Classifier')


    # ax1.set_ylim([0.05, 1.05])
    # ax2.set_ylim([0.05, 1.05])

    # ax1.set_yticks(np.arange(0.05, 1.05, step=0.05))
    # ax2.set_yticks(np.arange(0.05, 1.05, step=0.05))

    # ax1.set_ylim([0.5, .97])
    # ax2.set_ylim([0.5, .97])

    # ax1.set_yticks(np.arange(0.5, .97, step=0.05))
    # ax2.set_yticks(np.arange(0.5, .97, step=0.05))

    fig.legend(prob_tasks_list, loc='upper right')
    fig.tight_layout()
    # plt.show()

    plt.savefig(path_save + task + '_v_vs_t_data-reliability_results.png')
    print(path_save + task + '_v_vs_t_data-reliability_results.png')


def line_plot_prob_tasks_v2(file_old, file_new, path_save, v_t):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12.8, 9.6))

    # Set the axes title font size
    plt.rc('axes', titlesize=22)  # Set the axes labels font size
    # Set the font size of the figure title
    plt.rc('figure', titlesize=25)

    ax_list = [ax1[0], ax1[1], ax2[0], ax2[1]]

    x_axis = file_old.index.to_list()
    # x_t_axis = file_old.index.to_list()

    # colors_list = ["royalblue", "darkorange"]  #, "green", "red", "gold", "turquoise", "magenta"]
    colors_list_f = ['#f9d030', '#b8ee30', '#26dfd0', '#f62aa0']
    colors_list = sns.color_palette('colorblind', 8)
    dot_symbols_list = ['o', 'v']  # , '*', 'x', '^', 'H', 'D']

    prob_tasks_list = sorted(file_new.keys())
    count = 0

    if file_new.columns[0].split('_')[0] == 'OBJ':
        task = 'obj_number'
    elif file_new.columns[0].split('_')[0] == 'SUBJ':
        task = 'subj_number'

    if v_t == 'v':
        v_t_color = colors_list[0]
        # new_color = colors_list[4]
        fig.suptitle('Data Reliability\n Task: ' + task + '\nLinear Regression Classifier\nVisual Model')
    else:
        v_t_color = colors_list[1]
        # new_color = colors_list[2]
        fig.suptitle('Data Reliability\n Task: ' + task + '\nLinear Regression Classifier\nText Model')

    for i, ax_elem, prob_task in zip([0,1,2,3], ax_list, prob_tasks_list):
        count += 1
        ax_elem.plot(x_axis, file_new[prob_task], linestyle='-', color=colors_list_f[i], marker='*', markersize=15)
        # ax_elem.plot(x_t_axis, file_t_new[prob_task], linestyle='-', color=colors_list[4], marker='H', markersize=15)
        ax_elem.plot(x_axis, file_old['avg'], linestyle='-', color=v_t_color, marker='o', markersize=15)
        # ax_elem.plot(x_t_axis, file_t['avg'], linestyle='-', color=colors_list[1], marker='v', markersize=15)

        ax_elem.set_xlabel('Layers', fontsize=18)
        ax_elem.set_ylabel('Probing Accuracy', fontsize=18)
        ax_elem.set_title(prob_task, fontsize="15")



    fig.legend(['Edge Case Test Data', 'Original Test Data'], fontsize="15")
    fig.tight_layout()
    # plt.show()

    plt.savefig(path_save + v_t + '_' + task + '_data-reliability_results.png')
    print(path_save + v_t + '_' + task + '_data-reliability_results.png')
    plt.close()


if __name__ == '__main__':
    # tasks_dict = pd.read_csv('tasks_server.csv', index_col=0)
    # tasks_dict = pd.read_csv('tasks_classifier_lokal.csv', index_col=0)
    with open('config_visrep.yml') as config:
        config_dict = yaml.load(config, Loader=yaml.FullLoader)

        path_scores = config_dict['lokal_path'] + config_dict['path_scores'] + '20220923/'
        # path_png = config_dict['lokal_path'] + config_dict['path_scores'] + 'pngs/'
        path_pdfs = config_dict['lokal_path'] + config_dict['path_scores'] + 'pdfs/'
        filenames_test = natsorted(next(walk(path_scores), (None, None, []))[2])

        path_old_scores = config_dict['lokal_path'] + config_dict['data_path_in_sent']

        task_list = [('OBJ', 'obj_number'), ('SUBJ', 'subj_number')]  # , 'OBJ']
        # task_list = [('SUBJ', 'subj_number')]  # , 'OBJ']
        data_size = 40

        create_class_results = False  # True
        create_plots = True

        for task, task_name in task_list:
            print(task, task_name)
            if create_class_results:
                task_name = path_in_test.split('/')[-2]
                file_o_S_s_P = tasks_dict[task]['o_S_s_P']
                file_o_S_s_S = tasks_dict[task]['o_S_s_S']
                file_o_P_s_S = tasks_dict[task]['o_P_s_S']
                file_o_P_s_P = tasks_dict[task]['o_P_s_P']
                files_list = [file_o_S_s_S, file_o_S_s_P, file_o_P_s_S, file_o_P_s_P]
                # files_list = [file_o_S_s_S]

                path_out = tasks_dict[task]['path_out']
                path_scores = tasks_dict[task]['path_scores']
                path_old_scores = tasks_dict[task]['path_old_scores_10k']

                collect_test_scores = defaultdict()

                for m_type in ('t', 'v'):
                    for file in files_list:
                        print('file', file)
                        RunVisrep = VisRepEncodings(path_in_test + file, path_out)

                        if m_type == 'v':
                            RunVisrep.make_vis_model(m_type)
                        else:
                            RunVisrep.make_text_model(m_type)

                        # load saved models
                        classifier_models = natsorted(next(walk(path_out + m_type + '/10000_sav/'), (None, None, []))[2])
                        classifier_layers = [file.split('_')[1] for file in classifier_models]
                        classifier_path = path_out + m_type + '/10000_sav/'
                        data_test, scores_test_set = RunVisrep.load_classifier_model(data_size, classifier_models,
                                                                                     classifier_path)

                        data_test.to_csv(path_scores + 'data_test_' + file)

                        collect_test_scores[file] = scores_test_set

                    # directory = os.getcwd()
                    df_classifier = pd.DataFrame.from_dict(collect_test_scores)
                    df_classifier.to_csv(path_scores + m_type + '_' + task + '_' + str(data_size) + '.csv')

            if create_plots:
                # print(path_scores + 'v_' + task + '_' + str(data_size) + '.csv')
                print('\n Creating plots...\n')

                df_v_new = pd.read_csv(path_scores + 'v_' + task + '_' + str(data_size) + '.csv', index_col=0)
                df_t_new = pd.read_csv(path_scores + 't_' + task + '_' + str(data_size) + '.csv', index_col=0)

                df_v_old = pd.read_csv(path_old_scores + task_name + '/v/v_' + task + '_' + str(10000) + '.csv', index_col=0)
                df_t_old = pd.read_csv(path_old_scores + task_name + '/t/t_' + task + '_' + str(10000) + '.csv', index_col=0)

                # 'eval_SUBJ_train_train_raw_sentences.npy'
                train_subj_obj_dict = read_in_pickled_dict(path_old_scores + task_name + '/eval_' + task + '_train_train_raw_sentences.npy')
                train_subj_obj_dict_sum = sum(train_subj_obj_dict.values())
                print('train_subj_obj_dict_sum', train_subj_obj_dict_sum)
                test_subj_obj_dict = read_in_pickled_dict(path_old_scores + task_name + '/eval_' + task + '_test_test_raw_sentences.npy')
                test_subj_obj_dict_sum = sum(test_subj_obj_dict.values())
                print('train_subj_obj_dict_sum', test_subj_obj_dict_sum)
                # break

                df_header = list(df_t_new.columns)

                df_header_adapted = [name.split('3')[0] + '-train_'
                                     + str("{:.2f}".format((train_subj_obj_dict[name.split('3')[0].split('BJ_')[1]] /
                                                           train_subj_obj_dict_sum) * 100.0)) + '%'
                                     + '-test_'
                                     + str("{:.2f}".format((test_subj_obj_dict[name.split('3')[0].split('BJ_')[1]] /
                                                            test_subj_obj_dict_sum) * 100.0)) + '%'
                                     for name in df_header]
                # print('df_header_adapted', df_header_adapted)

                for df_n in [df_t_new, df_v_new]:
                    # print('df_n.columns', df_n.columns, '\n', df_header_adapted)
                    for name in df_header_adapted:
                        df_n.rename({str(name.split('-')[0] + '3.txt'): name}, errors="raise", axis=1, inplace=True)

                print('df_t_new.columns', df_t_new.columns)
                print('df_t_old.columns', df_t_old['avg'])
                # print('df_v_new.columns', df_v_new.columns)

                # build_key =
                # print(train_subj_obj_dict[])

                # for file in files_list:
                # sub_plots(task, path_scores, df_v_new, df_t_new, data_size / 2)
                # stack_plots(task, path_scores, df_v_new, df_t_new, df_v_old, df_t_old, data_size / 2)
                # line_plot_prob_tasks_v2(task, path_scores, df_v_new, df_t_new, df_v_old, df_t_old, data_size / 2)

                line_plot_prob_tasks_v1(df_v_old, df_v_new, df_t_old, df_t_new, path_pdfs)
                line_plot_prob_tasks_v2(df_v_old, df_v_new, path_pdfs, 'v')
                line_plot_prob_tasks_v2(df_t_old, df_t_new, path_pdfs, 't')

                # TODO: CREATE/ADPAT PLOT FUNCTION FOR SUBJ/OBJ TASK!!