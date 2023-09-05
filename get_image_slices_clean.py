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
    print('word: ', word.strip())
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

                 'è': 8, 'á': 7, 'à': 7, 'ó': 8, 'ò': 8, '€': 8, 'ô': 8, 'â': 7, 'ê': 8, '\'': 3, 'í': 3, '°': 6,
                 'ð': 8, 'ş': 6, 'Ş': 7, 'İ': 5, 'ç': 6, 'ý': 7, '@': 12, 'ě': 8, 'ī': 3, 'ź': 6, 'â': 7, 'ė': 8,
                 'î': 3, '«': 7, 'ï': 3, 'ñ': 8, 'ê': 8, 'и': 9, 'ά': 8, 'ш': 12, 'ū': 8, '÷': 8, 'ë': 8, 'δ': 8,
                 'É': 7, 'ð': 8, 'ď': 8, '»': 7, 'т': 6, '¾': 10, 'ň': 8, 'ø': 8, 'Í': 5, 'ς': 6,
                 'к': 7, 'Â': 9, 'č': 6, 'λ': 7, 'ổ': 8, 'ę': 8, 'Ş': 7, 'õ': 8, '®': 11, 'å': 8, 'Î': 5,
                 'í': 3, 'ő': 8, 'Ō': 10, 'Д': 9, 'Ç': 8, 'Á': 9, 'ɛ': 6, 'Ú': 10, 'κ': 7, 'İ': 5, 'Č': 8,
                 '‰': 16, '|': 7, 'ʿ': 2, 'έ': 6, 'µ': 8, '†': 7, 'ů': 8, '³': 5, 'ľ': 3, 'Γ': 7, 'Ţ': 7, '™': 10,
                 'ğ': 8, 'ś': 6, 'ă': 7, 'đ': 8, 'α': 8, '×': 8, '`': 4, 'л': 8, 'ι': 5, 'в': 8,
                 'Ц': 10, 'ί': 5, 'ʂ': 6, 'Σ': 8, 'ả': 7, 'ń': 8, 'н': 8, 'ã': 7, '²': 5, 'ć': 6, 'μ': 7, 'ı': 3,
                 '£': 8, 'ł': 3, 'ŏ': 8, 'ϊ': 5, 'ā': 7, 'Ś': 7, 'Š': 7, 'Ǩ': 8, 'Ω': 10, 'ş': 6,
                 'œ': 13, 'ţ': 5, 'њ': 12, 'ṫ': 5, 'ō': 8, 'š': 6, 'ω': 10, 'ř': 6, '·': 4, 'Ł': 7, 'ρ': 8,
                 '½': 10, 'Þ': 8, 'Ž': 8, 'π': 9, 'Æ': 12, 'ž': 6, 'ż': 6, 'ú': 8, 'σ': 8, 'æ': 12, 'ą': 7,

                 'ē': 8, 'ν': 7, '=': 8, 'М': 12, 'у': 7, 'ṯ': 5, 'ệ': 8, 'е': 8, 'ķ': 7, 'ο': 8, 'Е': 7,
                 'а': 7, 'Т': 7, 'ḥ': 8, 'с': 6, 'В': 9, 'О': 10, 'о': 8,

                 'ƒ': 8}

    # sent = 'test 123'
    # e.g.: {"Hier": [(0, 1.0), (1, 0.25)], ...]
    # sent_dict = defaultdict()
    sent = sentence.strip()
    print(sent)
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
        # end of word reached, but chars of new word can fit into current picture slice
        if char == ' ':
            count_pixels_in_word -= char_dict[char]
            count_chars_word -= 1
            # no char of new word yet processed but space already appeared
            if picture_slice < 30:
                if count_pixels_in_word != 0:  # chars_word != 0:
                    word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
                    if picture_slice - char_dict[' '] > 20:
                        mem_pixels_in_next_pic = picture_slice - 20 - char_dict[char]
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
            elif picture_slice >= 30:
                leftover_pic_slice = 10 - abs(30 - (picture_slice - char_dict[char]))
                char_mem_list = list()
                if count_pixels_in_word != 0:  # chars_word != 0:
                    word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
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

        # not end of word reached, but chars of new word will fit into current picture slice
        elif char != ' ':
            if picture_slice > 30:
                char_pixels_current_pic = (30 - (picture_slice - char_dict[char]))
                leftover_pic_slice = 10 - char_pixels_current_pic
                if char_pixels_current_pic >= char_dict[char]:
                    word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
                elif char_pixels_current_pic <= char_dict[char] and picture_slice - 20 >= count_pixels_in_word - 1:
                    pass
                else:
                    word_for_dict_list.append((pic_num, (count_pixels_in_word - char_dict[char] +
                                                         char_pixels_current_pic) / word_pixel))
                count_pixels_in_word = mem_pixels_in_next_pic + char_dict[char]
                mem_pixels_in_next_pic = 0
                picture_slice = picture_slice - 20
                char_mem = 0
                char_mem_list = list()
                pic_num += 1
            elif picture_slice == 30:
                char_pixels_current_pic = (30 - (picture_slice - char_dict[char]))
                leftover_pic_slice = 10 - char_pixels_current_pic
                if char_pixels_current_pic >= char_dict[char]:
                    word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
                elif char_pixels_current_pic <= char_dict[char] and picture_slice - 20 >= count_pixels_in_word - 1:
                    pass
                else:
                    word_for_dict_list.append((pic_num, (count_pixels_in_word - char_dict[char] +
                                                         char_pixels_current_pic) / word_pixel))
                if count_chars_word != len(word):
                    count_pixels_in_word = mem_pixels_in_next_pic + char_dict[char]
                else:
                    count_pixels_in_word = 0
                mem_pixels_in_next_pic = 0
                picture_slice = picture_slice - 20
                char_mem = 0
                char_mem_list = list()
                pic_num += 1

            elif picture_slice < 30:
                char_mem, char_mem_list = get_mem_val(char_mem, char_dict, char, char_mem_list)
                if picture_slice > 20 and picture_slice - char_dict[char] - 20 < 0:
                    mem_pixels_in_next_pic += picture_slice - 20
                elif picture_slice > 20 and picture_slice - char_dict[char] - 20 >= 0:
                    mem_pixels_in_next_pic += char_dict[char]

    if picture_slice != 0:
        diff_pic_slice = picture_slice - 30
        word_for_dict_list.append((pic_num, count_pixels_in_word / word_pixel))
    sent_list.append((word, word_for_dict_list, n_elem_pic_slice))

    return sent_list


def get_pic_num_for_word(sentence_list):
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
            clean_word_sent_list.append((word, sorted(collect_pic_nums)))

    return clean_word_sent_list

