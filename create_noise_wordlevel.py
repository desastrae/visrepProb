from modular_run_tr import VisRepEncodings
import os
import yaml
import sys


if __name__ == '__main__':
    with open('config_visrep.yml') as config, open('word_level/UD_dataset/UD_raw_sent_data.txt', 'w') as out:
        config_dict = yaml.load(config, Loader=yaml.FullLoader)

        if sys.argv[1] == 's':
            path_server_o_lokal = config_dict['server_path']
        elif sys.argv[1] == 'l':
            path_server_o_lokal = config_dict['lokal_path']
        else:
            print('Parameter ' + str(sys.argv[1]) + ' not existent.')
            exit(0)

        # path_server_o_lokal = config_dict['lokal_path']

        for task in config_dict['tasks_word'][:1]:
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

        RunVisrep = VisRepEncodings(config_dict, path_in_file, path_out, task, None)
        data = RunVisrep.read_UD_data(path_in_file)
        raw_sent_data = list()

        for sent_data in data[int(config_dict['dataset_size'][0] * 0.75):]:
            data_tuple_list = list(zip(*sent_data))
            raw_sent_data.append(' '.join(data_tuple_list[1]))

        out.write('\n'.join(raw_sent_data))

