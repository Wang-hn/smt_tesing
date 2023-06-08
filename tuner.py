"""
Copyright 2023 WHN

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random

from utils.strategy import StrategyEnumerator
import torch
from torch import nn
import timeout_decorator
import z3
import queue
import heapq
import json
import os
import re
from fastsmt.language import objects
from agent import SMTSolver


BV_THEORY = [
    'bvadd', 'bvsub', 'bvneg', 'bvmul', 'bvurem', 'bvsrem', 'bvsmod',
    'bvshl', 'bvlshr', 'bvashr', 'bvor', 'bvand', 'bvnand', 'bvnor',
    'bvxnor', 'bvule', 'bvult', 'bvugt', 'bvuge', 'bvsle', 'bvslt',
    'bvsge', 'bvsgt', 'bvudiv', 'extract', 'bvudiv_i', 'bvnot',
]

ST_TOKENS = [
    '=', '<', '>', '==', '>=', '<=', '=>', '+', '-', '*', '/',
    'true', 'false', 'not', 'and', 'or', 'xor',
    'zero_extend', 'sign_extend', 'concat', 'let', '_', 'ite',
    'exists', 'forall', 'assert', 'declare-fun', 
    'Int', 'Bool', 'BitVec',
]

ALL_TOKENS = ["UNK"] + ST_TOKENS + BV_THEORY


def uniq_list(lst):
    n_lst = []
    for lst_i in range(len(lst)):
        if lst_i == 0 or lst[lst_i] != lst[lst_i-1]:
            n_lst.append(lst[lst_i])
    return n_lst


class GoalTokenizer(object):
    def __init__(self):
        self.token_idx = {}
        for i, token in enumerate(ALL_TOKENS):
            self.token_idx[token] = i

    def bow(self, formula):
        if type(formula) is not str:
            formula = str(formula)
        digit = re.compile("\d+")
        formula = re.sub('[\(\)\n]', ' ', formula)
        formula = re.sub('\[|\]|\,', '', formula)
        formula = re.sub('[ ]+', ' ', formula)

        tokens = formula.split(' ')

        ret = [0 for i in ALL_TOKENS]
        for token in tokens:
            if token not in self.token_idx:
                ret[0] += 1
            else:
                ret[self.token_idx[token]] += 1
        return ret


import time

class Tuner:
    def __init__(self, config):
        self.enumerator = StrategyEnumerator(**config["tactics_config"])
        self.tokenizer = GoalTokenizer()
        self.solver = SMTSolver(self.tokenizer, self.enumerator)

    def solve(self, formula, tactic, use_rlimit=True):
        if type(tactic) is str:
            tactic = z3.Tactic(tactic)
        tactic = z3.TryFor(tactic, 5000)
        s = tactic.solver()

        s.add(formula)
        r_before = self.solver.get_rlimit(s)
        t_before = time.time()
        res = s.check()
        t_after = time.time()
        r_after = self.solver.get_rlimit(s)


        return str(res), s.assertions(), t_after-t_before if not use_rlimit else r_after-r_before

    def get_probes(self, formula):
        goal = z3.Goal()
        goal.add(formula)
        probes = [z3.Probe(p) for p in z3.probes()]
        ng = goal.as_expr()
        probes = [p(ng) for p in probes]
        return probes + self.tokenizer.bow(formula)

    def random_params(self, tactic):
        args = {tac : random.random() for tac in self.enumerator.param_max[tactic]}

        w_tac = self.enumerator.get_tactic_with_args(tactic, args)
        return w_tac

    def tuning(self, smt_instances, tac_sequence, cnt, quick_tuner=False):
        tsp = []
        

        def str_tactic_seq(x):
            return str([str(t) for t in x])


        def comp(x, y):
            sx = str_tactic_seq(x)
            sy = str_tactic_seq(y)
            if sx > sy: 
                return 1
            elif sx < sy: 
                return -1
            else:
                return 0
        
        from functools import cmp_to_key
        
        
        tac_sequence.sort(key=cmp_to_key(comp))
        n_tsp = [tac_sequence[0]]
        for i in range(len(tac_sequence)):
            if i > 0 and str_tactic_seq(tac_sequence[i]) != str_tactic_seq(tac_sequence[i-1]):
                n_tsp.append(tac_sequence[i])
        
        tac_sequence = n_tsp
        
        # cnt = min(cnt, len(tac_sequence))

        for seq in tac_sequence:
            seq = [objects.Tactic(tac) for tac in seq]
            tsp.append(seq)
            n_seq = [self.random_params(tac.s) for tac in seq]
            if str_tactic_seq(n_seq) == str_tactic_seq(seq):
                continue
            for i in range(10):
                tsp.append([self.random_params(tac.s) for tac in seq])

        tsp.sort(key=cmp_to_key(comp))
        # for i in tsp:
        #     print(str_tactic_seq(i))
        n_tsp = [tsp[0]]
        for i in range(len(tsp)):
            if i > 0 and str_tactic_seq(tsp[i]) != str_tactic_seq(tsp[i-1]):
                n_tsp.append(tsp[i])

        tsp = n_tsp
        
        cnt = min(cnt, len(tsp))
        if quick_tuner or cnt == len(tsp):
            return tsp

        print("uniq tactic_seq: {}".format(len(tsp)))
        formulas = [z3.parse_smt2_file(smt_instance) for smt_instance in smt_instances]
        for ts in res:
            print([(x.s, x.params) if isinstance(x, objects.With) else x.s for x in ts])
        heap = []
        print("=================has {} method to use, shrink to {} method".format(len(res), cnt))
        for i, ts in enumerate(res):
            print("=======use method", i, "solve formula")
            tot_time = 0
            t_tac = objects.AndThen(*ts).tactic if len(ts) > 1 else ts[0].tactic
            for k, formula in enumerate(formulas):
                if k % 2 == 0:
                    print("try to solve", k, "th formula")
                
                rres, formula, time = self.solve(formula, t_tac)
                if rres is 'unknown':
                    tot_time += 5500000
                else:
                    tot_time += time
            heapq.heappush(heap, (tot_time, i))

        tsp = res
        
        res = []
        for i in range(min(cnt, len(heap))):
            _, ts = heapq.heappop(heap)
            res.append(tsp[ts])

        return res

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', type=str, default='experiments/configs/normal_config.json')
    parser.add_argument('--train_data', type=str, default='../experiments/data/coreutils/ttrain')
    parser.add_argument('--tactics', type=str, default='coreutils_exp5.gen.tac')
    parser.add_argument('--shrink_size', type=int, default=-1)
    parser.add_argument('--mode', type=str, default='tuner')
    parser.add_argument('--quick_tuner', type=bool, default=True)
    parser.add_argument('--out_file', type=str, default=None)

    args = parser.parse_args()

    data = []
    tuner = Tuner(json.load(open(args.configuration, 'r')))
    for root, directories, filenames in os.walk(args.train_data):
        for file in filenames:
            if file.endswith('.smt2'):
                data.append(os.path.join(root, file))

    import random
    tac_seq = []
    data = random.sample(data, int(len(data) / 4))

    from agent import parse_combine_tactic
    with open(args.tactics, 'r') as f:
        lines = f.readlines()
        for line in lines:
            s_line = line.split(' ')
            if len(s_line) == 2:
                tac = parse_combine_tactic(s_line[1])
                tmp = [tac.s] if isinstance(tac, objects.Tactic) else [t.s for t in tac.v]
                tmp = uniq_list(tmp)
                tac_seq.append(tmp)
        f.close()
    for tac in tac_seq:
        pass
    
    ts = tuner.tuning(data, tac_seq, len(tac_seq) if args.shrink_size < 0 else args.shrink_size, args.quick_tuner)
    for ls in ts:
        print([str(tt) for tt in ls])

    if args.out_file is not None:
        with open(args.out_file, 'w') as f:
            f.write(str([str(tt) for tt in ls]))
            f.write('\n')


if __name__ == '__main__':
    main()

