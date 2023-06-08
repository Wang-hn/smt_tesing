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
import argparse
import json
import math
import os

import z3

from combiner import Cond, ProbeCond
from language import objects

def isstrategy(obj):
    return isinstance(obj, (objects.AndThen, objects.Tactic, objects.With, Cond))


class SMTTransformer:
    def __init__(self):
        pass

    def transform(self, source, in_type, out_type='strategy'):
        src = None
        if in_type.startswith('file'):
            lines = self.read_file(source)
            in_type = in_type[5:]
            if in_type == 'list':
                src = self.parse_lists(lines)
            elif in_type == 'smt':
                src = self.parse_smts(lines)
            elif in_type == 'strategy':
                src = self.parse_strategys(lines)
        elif in_type == 'list':
            src = self.list2strategy([source])
        elif in_type == 'smt':
            src = self.parse_smts([source])

        if out_type == 'strategy':
            return src
        elif out_type == 'list':
            return self.strategy2list(src)
        return None

    def read_file(self, source):
        res = []
        with open(source, 'r') as f:
            res = f.readlines()
        return res

    def parse_list(self, lst):
        res = []
        for st in lst:
            res.append(self.parse_strategy(st))
        return objects.AndThen(*res) if len(res) > 1 else res[0] if len(res) > 0 else None

    def parse_lists(self, lines):
        res = []
        for line in lines:
            if line == '' or line == '[]':
                continue
            lst = eval(line)
            assert isinstance(lst, list), 'error type {}'.format(line)
            res.append(self.parse_list(lst))

        return res

    def parse_strategys(self, lines):
        res = []
        for line in lines:
            if line == '':
                continue
            res.append(self.parse_strategy(line))
        return res

    def list2strategy(self, source):
        res = []
        for tac_seq in source:
            if len(tac_seq) > 1:
                res.append(objects.AndThen(*tac_seq))
            elif len(tac_seq) > 0:
                res.append(res[0])
        return res

    def strategy2list(self, source):
        res = []
        for tac in source:
            if isinstance(tac, objects.AndThen):
                res.append(tac.v)
            else:
                res.append([tac])
        return res

    def parse_strategy(self, line):
        if line.startswith('Tactic'):
            pos1 = line.find('(')
            pos2 = line.find(')', pos1)
            return objects.Tactic(line[pos1 + 1: pos2])
        if line.startswith('Probe'):
            pos1 = line.find('(')
            pos2 = line.find(')')
            return objects.Probe(line[pos1 + 1: pos2])
        if line.startswith('With'):
            pos1 = line.find(';')
            tac1 = line[5:pos1]
            line = line[pos1 + 1:]
            dic = {}
            #print(line)
            while line.find(';') > 0:
                pos1 = line.find(';')
                pos2 = line.find('=')
                #print(line[:pos2], line[pos2+1:pos1], line[pos1:])
                dic[line[:pos2]] = eval(line[pos2+1:pos1])
                line = line[pos1+1:]
            pos1 = line.find('=')
            dic[line[:pos1]] = eval(line[pos1+1:line.find(')')])
            return objects.With(tac1, dic)
        if line.startswith('Cond'):
            balance = 0
            pos1 = line.find(',')
            pos2 = line.find('>')
            pb1 = self.parse_strategy(line[5: pos2 - 1])
            num1 = float(line[pos2 + 1: pos1])
            pb1 = combiner.ProbeCond(pb1, num1)

            line = line[pos1:]

            tac_v = []

            for i in range(len(line)):
                if line[i] == ' ' and balance == 0:
                    pos2 = i + 1
                if line[i] == '(':
                    balance += 1
                if line[i] == ')':
                    balance -= 1
                    if balance == 0:
                        tac_v.append(self.parse_strategy(line[pos1: i + 1]))
            return combiner.Cond(pb1, tac_v[0], tac_v[1])
        if line.startswith('AndThen'):
            balance = 0
            line = line[8:]
            tac_v = []
            pos1 = 0
            for i in range(len(line)):
                if line[i] == ',' and balance == 0:
                    pos1 = i + 1
                if line[i] == '(':
                    balance += 1
                if line[i] == ')':
                    balance -= 1
                    if balance == 0:
                        tac_v.append(self.parse_strategy(line[pos1: i + 1]))

            return objects.AndThen(*tac_v)
        if line.startswith('['):
            pos1, pos2 = 0, 0
            tac_v = []
            while True:
                pos1 = line.find('\'', pos2+1)
                if pos1 == -1:
                    break
                pos2 = line.find('\'', pos1+1)
                tac_v.append(self.parse_strategy(line[pos1+1:pos2]))
            return objects.AndThen(*tac_v) if len(tac_v) > 1 else tac_v[0]
        return None

    def parse_smts(self, source):
        res = []
        for line in source:
            if line == '':
                continue
            res.append(self.parse_smt(line))
        return res

    def parse_smt(self, source):
        if not isinstance(source, str):
            return None

        if source.startswith('('):
            source = self.elim_brackets(source)

        # print(source)

        if source.startswith('>'):
            word, pos = self.find_next_space(source)
            word1, pos = self.find_next_space(source, pos)
            word2, pos = self.find_next_space(source, pos)
            # print(word, word1, word2)
            return ProbeCond(objects.Probe(word1), eval(word2))

        if source.startswith('if'):
            cond, pos = self.find_next_brackets(source)
            cond = self.parse_smt(cond)
            if source[pos] == '(' or source[pos+1] == '(':
                st_a, pos = self.find_next_brackets(source, pos)
            else:
                st_a, pos = self.find_next_space(source, pos+1)
            if source[pos] == '(' or source[pos+1] == '(':
                st_b, _ = self.find_next_brackets(source, pos)
            else:
                st_b, _ = self.find_next_space(source, pos+1)
            st_a = self.parse_smt(st_a)
            st_b = self.parse_smt(st_b)
            prefix, st_a, st_b = self.pop_prefix(st_a, st_b)

            if st_a is None:
                if st_b is None:
                    return objects.AndThen(*prefix) if len(prefix) > 1 else prefix[0]
                else:
                    return objects.AndThen(*prefix, st_b)
            if st_b is None:
                return objects.AndThen(*prefix, st_a)

            st_cb = Cond(cond, st_a, st_b)
            # print(*prefix, str(st_a), str(st_b))
            return objects.AndThen(*prefix, st_cb) if len(prefix) > 0 else st_cb

        if source.startswith('using-params'):
            word, pos = self.find_next_space(source)
            word, pos = self.find_next_space(source, pos)
            params = {}
            while pos is not None:
                word1, pos = self.find_next_space(source, pos)
                word2, pos = self.find_next_space(source, pos)
                if word2 == 'false':
                    word2 = 'False'
                if word2 == 'true':
                    word2 = 'True'
                params[word1[1:]] = eval(word2)
            
            return objects.With(word, params)

        if source.startswith('then'):
            word, pos = self.find_next_space(source)
            wt = []
            while pos is not None:
                if source[pos] == ' ':
                    pos+=1
                if source[pos] == '(':
                    word, pos = self.find_next_brackets(source, pos)
                else:
                    word, pos = self.find_next_space(source, pos)
                tac = self.parse_smt(word)
                if isinstance(tac, objects.AndThen):
                    wt += tac.v
                else:
                    wt.append(tac)
            return objects.AndThen(*wt) if len(wt) > 1 else wt[0]
        return objects.Tactic(source)

    def pop_prefix(self, st_a, st_b):
        st_a = st_a.v if isinstance(st_a, objects.AndThen) else [st_a]
        st_b = st_b.v if isinstance(st_b, objects.AndThen) else [st_b]

        res = []

        for i in range(min(len(st_a), len(st_b))):
            if str(st_a[i]) == str(st_b[i]):
                res.append(st_a[i])
            else:
                # something wrong?
                break

        st_a = st_a[len(res):]
        st_b = st_b[len(res):]

        st_a = objects.AndThen(*st_a) if len(st_a) > 1 else st_a[0] if len(st_a) > 0 else None
        st_b = objects.AndThen(*st_b) if len(st_b) > 1 else st_b[0] if len(st_b) > 0 else None

        return res, st_a, st_b


    def elim_brackets(self, source):
        if source[0] != '(':
            return
        balance = 0
        for i in range(len(source)):
            if source[i] == '(':
                balance += 1
            elif source[i] == ')':
                balance -= 1
                if balance == 0:
                    return source[1:i]
        return ''

    def find_next_brackets(self, source, pos=0):
        st = source.find('(', pos)
        if st < 0:
            return '', None
        balance = 0
        for i in range(st, len(source)):
            if source[i] == '(':
                balance += 1
            elif source[i] == ')':
                balance -= 1
                if balance == 0:
                    return source[st:i+1], i+1 if i+1<len(source) else None
        return '', None

    def find_next_space(self, source, pos=0):
        st = source.find(' ', pos)
        if st < 0:
            return source[pos:], None
        return source[pos:st], st+1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='shorten')
    parser.add_argument('--tactics', type=str)
    parser.add_argument('--out_type', type=str, default=None)
    parser.add_argument('--add_prefix', type=str, default=None)
    args = parser.parse_args()

    tf = SMTTransformer()

    if args.mode == 'shorten' or args.mode == 'smt':
        tacs = tf.transform(args.tactics, 'file_smt', 'strategy')[0]
        if args.add_prefix is not None:
            tacs = objects.AndThen(objects.Tactic(args.add_prefix), tacs)
        if args.mode is not None:
            print(str(tacs) if args.out_type == 'strategy' else tacs.to_smt2())
        else:
            print(tacs.to_smt2())
    elif args.mode == 'list':
        tacs = tf.transform(args.tactics, 'file_list')
        for tac in tacs:
            if args.out_type is not None:
                print(str(tac) if args.out_type == 'strategy' else tacs.to_smt2())
            else:
                print(str(tac))
    elif args.mode == 'strategy':
        tacs = tf.transform(args.tactics, 'file_strategy', out_type='list')
        res = set()
        for tac in tacs:
            res.add(str([str(t) for t in tac]))
        for tac in res:
            print(tac)

if __name__ == '__main__':
    main()
