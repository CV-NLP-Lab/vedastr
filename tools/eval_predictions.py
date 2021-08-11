import argparse
import copy
import Levenshtein
import os
import pathlib
import sys

def get_dict(filename, alphabet="'-.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    with open(filename) as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        l = line.strip('\n')
        if len(l) == 0:
            continue
        img_path, label = l.split('\t')[:2]
        if not set(label.lower()).issubset(alphabet):
            continue
        word_dict[pathlib.PurePath(img_path).name] = label
    return word_dict

def detailed_stat(predictions_file, target_file, long_word_min_len=10, alphabet="'-.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    pred_dict = get_dict(predictions_file, alphabet)
    target_dict = get_dict(target_file, alphabet)
    lower_alphabet = ''.join(set(alphabet.lower()))

    cases = ('LC + UC, all words', 'LC + UC, long words', 'MC, all words', 'MC, long words')
    matrix1 = {'del': {char: 0 for char in alphabet}, 'ins': {char: 0 for char in alphabet}}
    matrix1 = dict(**matrix1, **{sym: {char: 0 for char in alphabet} for sym in alphabet})
    matrix2 = {'del': {char: 0 for char in lower_alphabet}, 'ins': {char: 0 for char in lower_alphabet}}
    matrix2 = dict(**matrix2, **{sym: {char: 0 for char in lower_alphabet} for sym in lower_alphabet})
    matrices = {case: copy.deepcopy(matrix1) if case[0] == 'L' else copy.deepcopy(matrix2) for case in cases}

    if alphabet == alphabet.lower():
        cases = cases[2:]

    mistakes = []
    for name, target in target_dict.items():
        pred = pred_dict[name]
        pred_target = {'MC': (pred.lower(), target.lower())}
        if alphabet != alphabet.lower():
            pred_target['LC + UC'] = (pred, target)
        else:
            pred, target = pred.lower(), target.lower()

        if target != pred:
            mistakes.append((pred, target, name))

        is_long = (len(target) >= long_word_min_len)

        for case, (pred, target) in pred_target.items():
            ops = Levenshtein.editops(pred, target)
            matrices_names = [case + ', all words',]
            if is_long:
                matrices_names.append(case + ', long words')
            for matrix_name in matrices_names:
                for op in ops:
                    if op[0] == 'insert':
                        matrices[matrix_name]['ins'][target[op[2]]] += 1
                    elif op[0] == 'delete':
                        matrices[matrix_name]['del'][pred[op[1]]] += 1
                    elif op[0] == 'replace':
                        matrices[matrix_name][pred[op[1]]][target[op[2]]] += 1
    
    for case in cases:
        print(case + ':')
        matrix = matrices[case]
        first = '\t' + '\t'.join(matrix['ins'].keys())
        print(first)
        for key in matrix.keys():
            row = matrix[key]
            line = key + '\t'
            for sym in matrix['ins'].keys():
                line += '{}\t'.format(row[sym])
            print(line)
        print()
        print()

def general_stat(predictions_file, target_file, long_word_min_len=10, alphabet="'-.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    pred_dict = get_dict(predictions_file, alphabet)
    target_dict = get_dict(target_file, alphabet)

    cases = ('LC + UC, all words', 'LC + UC, long words', 'LC + UC, short words',
             'MC, all words', 'MC, long words', 'MC, short words')
    if alphabet == alphabet.lower():
        cases = cases[3:]
        
    correct_num = {case: 0 for case in cases}
    edit_dis = {case: 0.0 for case in cases}
    norm_edit_dis = {case: 0.0 for case in cases}
    long_num = 0
    short_num = 0
    all_num = 0
    long_char = 0
    short_char = 0
    all_char = 0

    for name, target in target_dict.items():
        pred = pred_dict[name]
        pred_target = {'MC': (pred.lower(), target.lower())}
        if alphabet != alphabet.lower():
            pred_target['LC + UC'] = (pred, target)

        is_long = (len(target) >= long_word_min_len)

        for case, (pred, target) in pred_target.items():
            cur_edit_dis = Levenshtein.distance(pred, target)
            is_correct = int(pred == target)

            correct_num[case + ', all words'] += is_correct
            edit_dis[case + ', all words'] += cur_edit_dis
            norm_edit_dis[case + ', all words'] += cur_edit_dis / len(target)
            
            suf = ', short words'
            if is_long:
                suf = ', long words'
            correct_num[case + suf] += is_correct
            edit_dis[case + suf] += cur_edit_dis
            norm_edit_dis[case + suf] += cur_edit_dis / len(target)

        all_char += len(target)
        all_num += 1

        if is_long:
            long_char += len(target)
            long_num += 1
        else:
            short_char += len(target)
            short_num += 1
            
    case_function = lambda case, all_, short, long: max(all_ if (case[-9:-6] == 'all') else (long if (case[-10:-6] == 'long') else short), 1)
    acc = {'Accuracy ({})'.format(case): 1.0 * correct_num / case_function(case, all_num, short_num, long_num)
            for case, correct_num in correct_num.items()}
    norm_edit_dis = {'Normalized edit distance 1 ({})'.format(case): dis / case_function(case, all_num, short_num, long_num)
            for case, dis in norm_edit_dis.items()}
    edit_dis = {'Normalized edit distance 2 ({})'.format(case): dis / case_function(case, all_char, short_char, long_char)
            for case, dis in edit_dis.items()}
    quality_dict = dict(**acc, **norm_edit_dis, **edit_dis)

    return quality_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--long_len', default=10, help='Minimum length of a long word', type=int)
    parser.add_argument('-m', '--mode', default='general', help='general or full')
    parser.add_argument('-a', '--alphabet', default="'-.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument('predictions_file')
    parser.add_argument('target_file')
    args = parser.parse_args()
    if args.mode == 'full':
        detailed_stat(args.predictions_file, args.target_file, args.long_len, alphabet=args.alphabet)
    else:
        quality_dict = general_stat(args.predictions_file, args.target_file, args.long_len, alphabet=args.alphabet)
        for key, value in quality_dict.items():
            print(f'{key}: {value}')

