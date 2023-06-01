# load libraries
import torch
import fairseq
import numpy as np

# load visrep model, display images
from fairseq.models.visual import VisualTextTransformerModel
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
from typing import Union, List


def read_pos_raw_data(path_file_name):
    pos_tags_file_list = list()
    sentence_list = list()

    # with open('de-utb/de-train.tt') as file:
    with open(path_file_name) as file:
        for line in file:
            if line == '\n':
                pos_tags_file_list.append(sentence_list)
                sentence_list = list()
                if len(pos_tags_file_list) == 20:
                    break
                continue
            sentence_list.append(tuple(line.split()))
    return pos_tags_file_list


def show_images(images: List[np.ndarray]) -> None:
    n: int = len(images)
    f = plt.figure(figsize=(20, 5))
    for i in range(n):
        # token figure
        ax = f.add_subplot(1, n, i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show(block=True)


def get_mem_val(char_mem, char_dict, char, char_mem_list):
    test_val = char_mem + char_dict[char]
    if test_val <= 10:
        char_mem += char_dict[char]
        char_mem_list.append(char)
        return char_mem, char_mem_list
    else:
        try:
            char_mem -= char_dict[char_mem_list.pop(0)]
            char_mem += char_dict[char]
            char_mem_list.append(char)
            if char_mem > 10:
                char_mem = char_dict[char]
                char_mem_list = [char]
                return char_mem, char_mem_list
            else:
                return char_mem, char_mem_list
        except IndexError:
            char_mem += char_dict[char]
            char_mem_list.append(char)
            if char_mem > 10:
                char_mem = char_dict[char]
                char_mem_list = [char]
                return char_mem, char_mem_list
            else:
                return char_mem, char_mem_list


def get_pixels_in_word(word, chardict):
    word_pixels = 0
    for char in word:
        word_pixels += chardict[char]
    return word_pixels


def get_wordpixels_in_pic_slice(sentence):
    char_dict = {'m': 12, 'W': 12, 'M': 12, '%': 11, '&': 10, 'w': 10, 'Q': 10, 'O': 10, 'Ö': 10, 'G': 10, 'N': 10,
                 'U': 10, 'Ü': 10, 'H': 10, 'D': 10, '#': 9, 'Ä': 9, 'A': 9, 'B': 9, 'n': 8, 'u': 8, 'o': 8, 'p': 8,
                 'ü': 8, 'd': 8, 'b': 8, 'g': 8, 'h': 8, 'ö': 8, '~': 8, 'ß': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8,
                 '7': 8, '8': 8, '9': 8, '0': 8, 'R': 8, 'Z': 8, 'P': 8, 'K': 8, '<': 8, '>': 8, 'q': 8, 'e': 8, '+': 8,
                 '$': 8, 'Y': 8, 'X': 8, 'V': 8, 'C': 8, '1': 8, 'a': 7, 'k': 7, 'ä': 7, 'y': 7, 'x': 7, 'v': 7, '§': 7,
                 '*': 7, 'E': 7, 'L': 7, 'F': 7, 'S': 7, 'T': 7, '_': 6, 'z': 6, 's': 6, 'c': 6, '?': 6, 'r': 6, '{': 5,
                 '}': 5, 'f': 5, 't': 5, 'I': 5, '"': 5, '/': 5, '\\': 5, 'J': 4, '!': 4, ';': 4, ',': 4, '.': 4,
                 ':': 4, '-': 4, '[': 4, ']': 4, '(': 4, ')': 4, 'j': 3, 'i': 3, 'l': 3, ' ': 3, 'start': 3, 'é': 8,
                 'è': 8, 'á': 7, 'à': 7}

    # sent = 'test 123'
    # e.g.: {"Hier": [(0, 1.0), (1, 0.25)], ...]
    # sent_dict = defaultdict()
    sent = sentence.strip()
    # sent = "Hier ist eine Katze ."
    sent_list = list()

    # count slices of pictures for a given sentence
    pic_num = 0
    # counts up to 30
    picture_slice = char_dict['start']
    # reevaluate start val
    start_val = True
    # list of all words in a sentence, separated by space
    word_list = sent.split()
    # e.g. for "Hier": [(0, 1.0), (1, 0.25)]
    word_for_dict_list = list()
    # get word from sentence
    word = word_list.pop(0)
    # amount of pixels in word
    word_pixel = get_pixels_in_word(word, char_dict)
    # count how many characters in word are in picture slice
    count_chars_word = 0
    # count pixels processed
    count_pixels_in_word = 0
    # count pixels of word in next picture slice
    mem_pixels_in_next_pic = 0
    # keep a list of chars of size <= 10
    char_mem_list = list()
    # count chars of size <= 10
    char_mem = 0
    # amount of n pic-slices for current word
    n_elem_pic_slice = 1 if word_pixel // 20 == 0 else word_pixel // 20 + 1

    print('========' + word + '========')

    for char in sent:
        # determine start-val for picture-slices in beginning of sentence
        if start_val and char in 'HmMNUÜDBnuüpbhßRPK1kELFr[il':
            start_val = False
            picture_slice = 2
        elif start_val and char in '_j':
            start_val = False
            picture_slice = 4
        elif start_val and char in 'J':
            start_val = False
            picture_slice = 5
        else:
            start_val = False

        count_pixels_in_word += char_dict[char]
        count_chars_word += 1
        picture_slice += char_dict[char]
        print('word: ', word, 'char: ', char, 'char_dict[char]: ', char_dict[char], 'char_mem: ', char_mem,
              'char_mem_list: ', char_mem_list, 'count_chars_word: ', count_chars_word, 'picture_slice: ', picture_slice,
              'pic_num: ', pic_num, 'word_for_dict_list: ', word_for_dict_list, '\nsent_list', sent_list,
              '\nword_pixel: ', word_pixel, 'count_pixels_in_word: ', count_pixels_in_word, 'mem_pixels_in_next_pic: ',
              mem_pixels_in_next_pic)

        # end of word reached, but chars of new word can fit into current picture slice
        if char == ' ':
            count_pixels_in_word -= char_dict[char]
            count_chars_word -= 1
            print("space char", 'count_pixels_in_word', count_pixels_in_word)
            # no char of new word yet processed but space already appeared
            if picture_slice < 30:
                print("pic_slice < 30")
                if count_pixels_in_word != 0:  # chars_word != 0:
                    # print('bool(word_for_dict_list)', bool(word_for_dict_list))
                    # if count_chars_word != len(word) and bool(word_for_dict_list):
                    word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
                    print('word_for_dict_list', word_for_dict_list)
                    if picture_slice - char_dict[' '] > 20:
                        print("pic_slice > 20")
                        mem_pixels_in_next_pic = picture_slice - 20 - char_dict[char]
                        print('leftover_pic_slice', mem_pixels_in_next_pic, picture_slice)
                        # if mem_pixels_in_next_pic != 0:
                        #     word_for_dict_list.append((pic_num + 1, mem_pixels_in_next_pic / word_pixel))
                    sent_list.append((word, word_for_dict_list, n_elem_pic_slice))
                    word = word_list.pop(0)
                    word_pixel = get_pixels_in_word(word, char_dict)
                    n_elem_pic_slice = 1 if word_pixel // 20 == 0 else word_pixel // 20 + 1
                    count_pixels_in_word = 0
                    count_chars_word = 0
                    char_mem = 0
                    char_mem_list = list()
                    mem_pixels_in_next_pic = 0
                    word_for_dict_list = list()
                elif count_pixels_in_word == 0:
                    sent_list.append((word, word_for_dict_list, n_elem_pic_slice))
                    word = word_list.pop(0)
                    word_pixel = get_pixels_in_word(word, char_dict)
                    n_elem_pic_slice = 1 if word_pixel // 20 == 0 else word_pixel // 20 + 1
                    count_pixels_in_word = 0
                    count_chars_word = 0
                    char_mem = 0
                    char_mem_list = list()
                    mem_pixels_in_next_pic = 0
                    word_for_dict_list = list()
                print('========' + word + '========')
            elif picture_slice >= 30:
                print("pic_slice >= 30")
                leftover_pic_slice = 10 - abs(30 - (picture_slice - char_dict[char]))
                print('leftover_pic_slice', leftover_pic_slice, picture_slice)
                char_mem_list = list()
                if count_pixels_in_word != 0:  # chars_word != 0:
                    print('test')
                    # if count_chars_word != len(word):
                    word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
                    # if mem_pixels_in_next_pic != 0:
                        # word_for_dict_list.append((pic_num + 1, mem_pixels_in_next_pic / word_pixel))

                    sent_list.append((word, word_for_dict_list, n_elem_pic_slice))
                    word = word_list.pop(0)
                    word_pixel = get_pixels_in_word(word, char_dict)
                    n_elem_pic_slice = 1 if word_pixel // 20 == 0 else word_pixel // 20 + 1
                    count_pixels_in_word = 0
                    count_chars_word = 0
                    mem_pixels_in_next_pic = 0
                    char_mem = 0
                    char_mem_list = list()
                    word_for_dict_list = list()
                pic_num += 1
                picture_slice = picture_slice - 20
                print('========' + word + '========')

        # not end of word reached, but chars of new word will fit into current picture slice
        elif char != ' ':
            print("not space char")
            if picture_slice > 30:
                print("pic_slice >= 30")
                print(picture_slice, pic_num)
                char_pixels_current_pic = (30 - (picture_slice - char_dict[char]))
                leftover_pic_slice = 10 - char_pixels_current_pic
                print('charpixels', char_pixels_current_pic, char_dict[char])
                if char_pixels_current_pic >= char_dict[char]:
                    word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
                elif char_pixels_current_pic <= char_dict[char] and picture_slice - 20 >= count_pixels_in_word - 1:
                    pass
                else:
                    word_for_dict_list.append((pic_num, (count_pixels_in_word - char_dict[char] +
                                                         char_pixels_current_pic) / word_pixel))
                print('leftover_pic_slice', leftover_pic_slice, picture_slice, char_pixels_current_pic, count_pixels_in_word)
                count_pixels_in_word = mem_pixels_in_next_pic + char_dict[char]
                mem_pixels_in_next_pic = 0
                picture_slice = picture_slice - 20
                print('leftover_pic_slice', leftover_pic_slice, picture_slice, char_pixels_current_pic)
                # picture_slice += leftover_pic_slice
                print('pixels word: ', char_pixels_current_pic, count_pixels_in_word, leftover_pic_slice, word_pixel,
                      count_pixels_in_word - leftover_pic_slice)
                # picture_slice += char_dict[char]
                # chars_word = len(char_mem_list) + 1
                char_mem = 0
                char_mem_list = list()

                print('picture_slice', picture_slice)
                # test how much is left of word
                print('len(word)', len(word))
                pic_num += 1
                print('pic_num', pic_num)
            elif picture_slice == 30:
                print("pic_slice >= 30")
                print(picture_slice, pic_num)
                char_pixels_current_pic = (30 - (picture_slice - char_dict[char]))
                leftover_pic_slice = 10 - char_pixels_current_pic
                print('charpixels', char_pixels_current_pic, char_dict[char])
                if char_pixels_current_pic >= char_dict[char]:
                    word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
                elif char_pixels_current_pic <= char_dict[char] and picture_slice - 20 >= count_pixels_in_word - 1:
                    pass
                else:
                    word_for_dict_list.append((pic_num, (count_pixels_in_word - char_dict[char] +
                                                         char_pixels_current_pic) / word_pixel))
                print('leftover_pic_slice', leftover_pic_slice, picture_slice, char_pixels_current_pic,
                      count_pixels_in_word)
                if count_chars_word != len(word):
                    count_pixels_in_word = mem_pixels_in_next_pic + char_dict[char]
                else:
                    count_pixels_in_word = 0
                mem_pixels_in_next_pic = 0
                picture_slice = picture_slice - 20
                print('leftover_pic_slice', leftover_pic_slice, picture_slice, char_pixels_current_pic)
                # picture_slice += leftover_pic_slice
                print('pixels word: ', char_pixels_current_pic, count_pixels_in_word, leftover_pic_slice, word_pixel,
                      count_pixels_in_word - leftover_pic_slice)
                # picture_slice += char_dict[char]
                # chars_word = len(char_mem_list) + 1
                char_mem = 0
                char_mem_list = list()

                print('picture_slice', picture_slice)
                # test how much is left of word
                print('len(word)', len(word))
                pic_num += 1
                print('pic_num', pic_num)

            elif picture_slice < 30:
                print("pic_slice < 30")
                # get value in char_memory
                char_mem, char_mem_list = get_mem_val(char_mem, char_dict, char, char_mem_list)
                # if picture_slice - char_dict[char] > 20:
                #     mem_pixels_in_next_pic += char_dict[char]
                # elif picture_slice - 20 > 0:
                if picture_slice > 20 and picture_slice - char_dict[char] - 20 < 0:
                    print('mem_pixels_in_next_pic: ', mem_pixels_in_next_pic)
                    mem_pixels_in_next_pic += picture_slice - 20
                elif picture_slice > 20 and picture_slice - char_dict[char] - 20 >= 0:
                    print('mem_pixels_in_next_pic: ', mem_pixels_in_next_pic)
                    mem_pixels_in_next_pic += char_dict[char]

    print('final steps')
    if picture_slice != 0:
        diff_pic_slice = picture_slice - 30
        # print(diff_pic_slice, count_pixels_in_word, word_pixel)
        word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
    # sent_dict[word] = word_for_dict_list
    sent_list.append((word, word_for_dict_list, n_elem_pic_slice))
    # print(sent_list)

    return sent_list


def get_pic_num_for_word(sentence_list):
    print("Create clean pic_num list...")
    clean_word_sent_list = list()
    for word, pic_num_list, n_pic_slices in sentence_list:
        collect_pic_nums = list()
        best_pic_num_list = sorted(pic_num_list, key=lambda x: x[1], reverse=True)
        if best_pic_num_list[0][1] > 0.95:
            clean_word_sent_list.append((word, best_pic_num_list[0][0]))
            continue
        else:
            for pic_num, val in best_pic_num_list:
                if val < 0.10:
                    continue
                else:
                    collect_pic_nums.append(pic_num)
            # if len(collect_pic_nums) > n_pic_slices:
                # clean_word_sent_list.append((word, sorted(collect_pic_nums[:n_pic_slices])))
            # else:
            clean_word_sent_list.append((word, sorted(collect_pic_nums)))

    print(clean_word_sent_list)
    return clean_word_sent_list


lang_pair = "tr_models/visual/WMT_de-en"
# lang_pair = "tr_models/visual/de-en"
mdl = VisualTextTransformerModel.from_pretrained(
    checkpoint_file=lang_pair+'/checkpoint_best.pt',
    target_dict=lang_pair+'/dict.en.txt',
    target_spm=lang_pair+'/spm.model',
    # target_spm=lang_pair+'/spm_en.model',
    src='de'
)
mdl.eval()  # disable dropout (or leave in train mode to finetune)
# sent = "Ich bin ja ein robustes Model"
# sent_len = len(sent)
# print(sent_len)
# translate, encoding = mdl.translate(sent)
# translate, encoding = mdl.translate(sent)
# print(translate, len(encoding['l1']))
# sent_image, image_tokens = mdl.image_generator.get_images(sent)
# display(Image.fromarray(sent_image))
# plt.imshow(sent_image)
# print(len(image_tokens))
# plt.imshow(sent_image)
# plt.show()
# show_images(image_tokens)

# alpahbet = "1akäyxv§*ELFST_zsc?r{}ftI\"/\\J!;,.:-[]()jil "
# alpahbet = "_"
# alaphabet_list = [*alpahbet]
# for char_n in alaphabet_list:
    # char_test_str = 'HHaZZZZ'
    # char_test_str = 'ZZZaKZZ'
    # char_test_str = char_n + 'ZZZ'
    # char_test_str = 'KKKYeKK'
    # sent_image, image_tokens = mdl.image_generator.get_images(char_test_str)
    # print(len(image_tokens))
    # show_images(image_tokens)

# pos_file_sentences = read_pos_raw_data('word_level/de-utb/de-train.tt')
pos_file_sentences = read_pos_raw_data('word_level/de-utb/stunde.tt')

for sent in pos_file_sentences[:20]:
    sent_list, pos_list = list(zip(*sent))
    print(sent_list)
    sentence = ' '.join(sent_list)
    # sentence = 'One hour later, we were finally given a table , which had not even been cleared off.'
    print(sentence)
    pic_num_words = get_pic_num_for_word(get_wordpixels_in_pic_slice(sentence))
    sent_image, image_tokens = mdl.image_generator.get_images(sentence.strip())
    show_images(image_tokens)

    # sent = "Das Das Das Das Das"
    # sent_test = "Hier ist eine"  # Katze."
    # sent_test = "test 123"
