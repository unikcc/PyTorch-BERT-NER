import numpy as np
from collections import defaultdict


class Metric:
    def __init__(self):
        # tp, recall(gold), precision(predict)
        self.tp, self.fn, self.fp = defaultdict(int), defaultdict(int), defaultdict(int)
        self.all_key = 'Weight'
    
    def add_instance(self, predict, gold):
        predict = tuple(list(map(tuple, predict)))
        gold = tuple(list(map(tuple, gold)))
        for pred in predict:
            if pred in gold:
                self.tp[pred[-1]] += 1
                self.tp[self.all_key] += 1
        for pred in predict:
            self.fp[pred[-1]] += 1
            self.fp[self.all_key] += 1
        for gd in gold:
            self.fn[self.all_key] += 1
            self.fn[gd[-1]] += 1
    
    def get_score(self):
        keys = [w for w in list(self.tp) if w != self.all_key]
        tp_mi = sum([self.tp[w] for w in keys])
        fp_mi = sum([self.fp[w] for w in keys])
        fn_mi = sum([self.fn[w] for w in keys])
        p_mi = tp_mi / fp_mi if fp_mi > 0 else 0
        r_mi = tp_mi / fn_mi if fn_mi > 0 else 0
        f_mi = 2 * p_mi * r_mi / (p_mi + r_mi) if p_mi + r_mi > 0 else 0

        tp_ma = [self.tp[w] for w in keys]
        fp_ma = [self.fp[w] for w in keys]
        fn_ma = [self.fn[w] for w in keys]
        p_ma = [w/z if z > 0 else 0 for w, z in zip(tp_ma, fp_ma)]
        r_ma = [w/z if z > 0 else 0 for w, z in zip(tp_ma, fn_ma)]
        f_ma = [2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(p_ma, r_ma)]
        p_ma, r_ma, f_ma = [np.mean(w) for w in (p_ma, r_ma, f_ma)]

        return  (p_mi, r_mi, f_mi), (p_ma, r_ma, f_ma)
    
    def report(self):
        lengths = 10
        keys = [w for w in list(self.tp) if w != self.all_key] + [self.all_key]
        res = []
        head = ['', 'precision', 'recall', 'f1-score', 'support']
        str_format = '{:>' + str(lengths) + '}'
        num_format = '{:>' + str(lengths) + '.4f}'
        res.append(''.join(map(str_format.format, head)))
        res.append('')
        # res.append(''.join([str_format.format(w) for w in head]))
        for key in keys:
            line = [str_format.format(key)]
            p = self.tp[key] / self.fp[key] if self.fp[key] > 0 else 0
            r = self.tp[key] / self.fn[key] if self.fn[key] > 0 else 0
            f = 2 * p * r / (p + r) if p + r > 0 else 0
            line += [num_format.format(w) for w in (p, r, f)] + [str_format.format(self.fn[key])]
            res.append(''.join(line))
        prf_mi, prf_ma = self.get_score()
        res = res[:-1] + [''] + res[-1:]
        line = [str_format.format('Micro-score')]
        line += [num_format.format(w) for w in prf_mi] + [str_format.format(self.fn[self.all_key])]
        res.append(''.join(line))
        line = [str_format.format('Macro-score')]
        line += [num_format.format(w) for w in prf_ma] + [str_format.format(self.fn[self.all_key])]
        res.append(''.join(line))
        res = '\n'.join(res)
        # line += [num_format.format(w) for w in (p, r, f)] + [str_format.format(self.fn[key])]
        nums = 40
        # print("-"*nums, "My Report:", "-"*nums)
        # print('\n'.join(res))
        # print("-"*nums, "My Report:", "-"*nums)
        return p, r, f, res

    def reset_result(self):
        self.tp, self.fn, self.fp = 0, 0, 0
        
