def open_bpe_file():
    bpe_list = list()
    with open('bpe_list.txt', 'r') as f:
        for line in f:
            x = line[:-1]
            bpe_list.append(x)
    # print('bpe_list: ', bpe_list)
    return bpe_list


def ref_bpe_word(sent):
    # print(sent)
    # sent = ['Sehr', 'gute', 'Beratung', ',', 'schnelle', 'Behebung', 'der', 'Probleme', ',', 'so', 'stelle', 'ich', 'mir',
    #         'Kundenservice', 'vor', '.']
    # words_list = ['▁S', 'ehr', '▁gute', '▁Beratung', '▁', ',', '▁schnell', 'e', '▁Be', 'hebung', '▁der', '▁Probleme', '▁',
    #               ',', '▁so', '▁', 'stelle', '▁ich', '▁mir', '▁K', 'und', 'en', 'serv', 'ice', '▁vor', '▁', '.']
    words_list = open_bpe_file()
    # print(words_list)
    words_list = [elem.strip('▁') for elem in words_list]
    # print('words_list: ', words_list)
    sent_val = len(sent)

    word_mem = sent.pop(0)
    word_ref_list = list()
    compound_word = ''
    positions_list = list()
    index_pos = -1

    for word in words_list:
        index_pos += 1
        if word_mem == word:
            word_ref_list.append((word_mem, index_pos))
            try:
                word_mem = sent.pop(0)
            except IndexError:
                continue
        elif word_mem == compound_word:
            word_ref_list.append((word_mem, positions_list))
            try:
                word_mem = sent.pop(0)
            except IndexError:
                continue
            compound_word = ''
            positions_list = list()
        elif word == '':
            continue
        else:
            compound_word += word
            positions_list.append(index_pos)
            if word_mem == compound_word:
                word_ref_list.append((word_mem, positions_list))
                try:
                    word_mem = sent.pop(0)
                except IndexError:
                    continue
                compound_word = ''
                positions_list = list()

    # print(sent_val, len(word_ref_list))

    if sent_val != len(word_ref_list):
        print('\nword_ref_list: ', word_ref_list)
        print('\nsent: ', sent)
    return word_ref_list
