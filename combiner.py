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

from agent import SMTSolver, GoalTokenizer, StrategyEnumerator
from language import objects


class Cond:
    def __init__(self, p, t1, t2):
        self.p = p
        self.t1 = t1
        self.t2 = t2
        self.tactic = z3.Cond(p.probe, t1.tactic, t2.tactic)

    def __str__(self):
        return 'Cond(%s,%s,%s)' % (str(self.p), str(self.t1), str(self.t2))

    def to_smt2(self):
        return '(if (%s) (then %s) (then %s))' % (self.p.to_smt2(), self.t1.to_smt2(), self.t2.to_smt2())


class ProbeCond:
    def __init__(self, probe: objects.Probe, cond):
        self.probe = (probe.probe > cond)
        self.cond = cond
        self.ori = probe

    def __call__(self, g):
        return self.probe(g)

    def __str__(self):
        return '{} > {}'.format(str(self.ori), self.cond)

    def to_smt2(self):
        return '> %s %s' % (self.ori.s, str(int(self.cond+0.5)))


def find_prefix(tac_seqs):
    prefix = []
    tac_list = []
    for i, tac_seq in tac_seqs:
        tac_list.append((i, tac_seq.v if isinstance(tac_seq, objects.AndThen) else tac_seq))

    for i in range(min([len(tac_seq) for tac_seq in tac_list])):
        now_tac = None
        for _, tac_seq in tac_list:
            if i >= len(tac_seq):
                now_tac = None
                break
            if now_tac is None:
                now_tac = tac_seq[i]
                continue
            elif str(now_tac) != str(tac_seq[i]):
                now_tac = None
                break

        if now_tac is None:
            break
        prefix.append(now_tac)

    begin = len(prefix)
    # print(begin)
    n_tac_list = []
    for i, tac_seq in tac_list:
        n_tac_seq = tac_seq[begin:]
        if len(n_tac_seq) > 0:
            n_tac_list.append((i, n_tac_seq))
    # print(tac_list)
    # tac_list = [(i, tac_seq[begin:]) for i, tac_seq in tac_list]
    return prefix, n_tac_list


def split_data(formula_data, probe):
    d_is, d_not = [], []
    for name, data in formula_data:
        g = z3.Goal()
        g.add(data)
        # print(str(probe), probe(g))
        if probe(g) > 0.5:
            d_is.append((name, data))
        else:
            d_not.append((name, data))
    return d_is, d_not


def shorten_tacs(tac_seqs, t_is):
    res = []
    for i, tac_seq in tac_seqs:
        if isinstance(tac_seq, objects.AndThen):
            tac_seq = tac_seq.v
        if str(t_is) == str(tac_seq[0]):
            tac_seq = tac_seq[1:]
            if len(tac_seq) > 0:
                res.append((i, tac_seq))
    return res


def choose_tac_with_prefix(tac_seqs, t_is):
    res = []
    for i, tac_seq in tac_seqs:
        if isinstance(tac_seq, objects.AndThen):
            tac_seq = tac_seq.v
        if str(t_is) == str(tac_seq[0]):
            res.append((i, tac_seq))
    return res


class Combiner:
    def __init__(self, solver: SMTSolver, cache_path=None):
        self.TIMEOUT_COST = 50000000
        self.solver = solver
        self.min_data_len = 10
        self.solve_cache = None
        self.probe_dict = {probe: set() for probe in z3.probes()}
        self.next_cache = {}
        self.r_cache = {}
        self.cache_path=cache_path

    def gen_predicts(self):
        predicts = []
        for probe, values in self.probe_dict.items():
            if len(values) <= 1:
                continue
            p = objects.Probe(probe)
            step = int(len(values)/16)
            if step > 0:
                for i in range(16):
                    predicts.append(ProbeCond(p, int(list(values)[step*i])))
                # predicts.append(ProbeCond(p, list(values)[step]))
                # predicts.append(ProbeCond(p, list(values)[step*3]))
            predicts.append(ProbeCond(p, (max(values)+min(values))/2.0))
        print("=====gen predicts success")
        return predicts

    def append_probe_dict(self, datas):
        for p in z3.probes():
            if self.probe_dict.get(p) is None:
                self.probe_dict[p] = set()
            for _, formula in datas:
                g = z3.Goal()
                g.add(formula)
                self.probe_dict[p].add(z3.Probe(p)(g))
        return self.gen_predicts()

    def remake_probe_dict(self, datas):
        self.probe_dict = {}
        return self.append_probe_dict(datas)

    def gen_strategy(self, datas, tacs, predicts=None):
        tacs = [(i, tac) for i, tac in enumerate(tacs)]
        if self.solve_cache is None:
            if self.cache_path is not None:
                self.load_cache(self.cache_path)
            else:
                self.solve_cache = {}
            datas = self.init_solve_cache(datas, tacs, (predicts is None))
            if predicts is None:
                predicts = self.append_probe_dict(datas)
            if len(self.solve_cache) != len(datas):
                self.load_cache(self.cache_path)
        return self.__gen_strategy(datas, tacs, predicts)

    def __gen_strategy(self, datas, tacs, predicts):
        print("==========gen_strategy with {} datas".format(len(datas)))
        if len(datas) < self.min_data_len:
            best_tac = ((len(datas)+1, 0, -1), None)
            n_data = []
            for name, data in datas:
                n_data.append(name)
            for tac_i, tac in tacs:
                tac = objects.AndThen(*tac) if len(tac)>1 else tac[0]
                unsolved, rlimit = self.solver.solve_dataset(n_data, tac.tactic)
                best_tac = min(best_tac, ((unsolved, rlimit, tac_i), tac))
            return best_tac[1]

        pre_ts, tacs = find_prefix(tacs)
        print("pre_ts found")
        if len(pre_ts) > 0:
            print("forward begin")
            datas = self.forward_data(datas, pre_ts)
            print("forward done")

        t_is = t_not = None
        d_is = d_not = []
        minc = predicts[0]

        dlist = {c: split_data(datas, c) for c in predicts}

        print("gen pre_ts as follow:")
        print(str(objects.AndThen('skip', 'skip', *pre_ts)))
        remake = 0
        ft_is, ft_not = None, None

        bst_tac = (1e20, objects.Tactic('skip'), objects.Tactic('skip'))

        while remake <= len(predicts):
            if remake == len(predicts):
                _, t_is, t_not = bst_tac
                if t_is.s == 'skip':
                    return objects.AndThen(*pre_ts) if len(pre_ts) > 1 else pre_ts[0]
                elif str(t_is) != str(t_not):
                    break
                datas = self.forward_data(datas, t_is)
                tp = self.remake_probe_dict(datas)
                if len(tp) > 0:
                    predicts = self.remake_probe_dict(datas)
                dlist = {c: split_data(datas, c) for c in predicts}
                print("update pre_ts")
                # print(str(objects.AndThen('skip', 'skip', *pre_ts)))
                pre_ts.append(t_is)
                tacs = shorten_tacs(tacs, t_is)
                print(tacs)
                remake = 0
                bst_tac = (1e20, objects.Tactic('skip'), objects.Tactic('skip'))
            clist = [self.cost(c, dlist[c][0], dlist[c][1], tacs) for c in predicts]
            if remake < len(clist):
                if remake > 0:
                    print("remake {}/{}".format(remake, len(clist)))
                # print([str(p) for p in predicts])
                for i in range(remake):
                    clist[clist.index(min(clist))] = 1e20
                idx = clist.index(min(clist))
                if clist[idx] == 1e20:
                    remake = len(predicts)
                    continue
                remake = remake+1

            minc = predicts[idx]
            d_is, d_not = dlist[minc]
            print("try minc {} with cost {}".format(str(minc), clist[idx]))
            
            sc_is, t_is = self.find_min_tac(d_is, tacs)
            print(sc_is, str(t_is))
            sc_not, t_not = self.find_min_tac(d_not, tacs)
            print(sc_not, str(t_not))

            r_is, r_not = len(d_is)/(len(datas)+1e-25), len(d_not)/(len(datas)+1e-25)
            if bst_tac[0] > r_is*sc_is+r_not*sc_not:
                bst_tac = (r_is*sc_is+r_not*sc_not, t_is, t_not)

            print("find tac branch {} and {}".format(t_is, t_not))

        if len(tacs) == 0:
            return objects.AndThen(*pre_ts) if len(pre_ts) > 1 else pre_ts[0]

        print("go to other branch {} and {}".format(str(t_is), str(t_not)))
        p_is = self.remake_probe_dict(d_is)
        p_not = self.remake_probe_dict(d_not)
        s_is = self.__gen_strategy(d_is, choose_tac_with_prefix(tacs, t_is), p_is)
        s_not = self.__gen_strategy(d_not, choose_tac_with_prefix(tacs, t_not), p_not)
        back_ts = Cond(minc, s_is, s_not)
        return objects.AndThen(*pre_ts, back_ts) if len(pre_ts) > 0 else back_ts

    def forward_data(self, old_data, tac_seq):
        if not isinstance(tac_seq, list):
            tac_seq = [tac_seq]
        if tac_seq is None:
            return old_data
        n_data = []
        cnt = 0
        stp = int(len(old_data) / 20) + 1
        for name, data in old_data:
            if cnt % stp == 0:
                print("forward {}th data".format(cnt))
            cnt = cnt + 1
            if isinstance(data, str):
                formula = z3.parse_smt2_file(data)
            else:
                formula = data

            # while len(tac_seq) > 0 and self.next_cache.get(str(data)) is not None and self.next_cache[str(data)].get(str(tac_seq[0])) is not None:
            #     formula = self.next_cache[str(data)][str(tac_seq[0])]
            #     tac_seq = tac_seq[1:]
            if len(tac_seq) > 0:
                _, _, n_formula, _ = self.solver.solve_with_tactic_seq(formula, tac_seq)
            else:
                n_data.append((name, formula))
                n_formula = formula
                continue
            if formula is not None and str(formula) != str(n_formula):
                n_data.append((name, n_formula))
        return n_data

    def cost(self, predict, d_is, d_not, tac_seqs):
        if len(d_is) == 0 or len(d_not) == 0:
            return float('inf')
        tot_len = len(d_is) + len(d_not)

        def hts(ds):
            tot = 0.0
            dname = [name for name, formula in ds]
            for i, tac_seq in tac_seqs:
                solved = len(set(dname) & set(self.solve_cache[i].keys()))
                ratio = solved / len(ds)
                if ratio > 0.5:
                    ratio -= 0.001
                else:
                    ratio += 0.001
                tot += ratio * math.log(ratio) + (1-ratio) * math.log(1-ratio)
            return -tot

        return len(d_is)/tot_len*hts(d_is) + len(d_not)/tot_len*hts(d_not)

    def find_min_tac(self, d_is, tac_seqs):
        res = (1e20, objects.Tactic('skip'))
        if len(tac_seqs) == 0:
            return res
        if len(d_is) == 0:
            tac_seq = tac_seqs[0][1]
            if len(tac_seq) > 0:
                return (1e19, tac_seqs[0][1][0])
            else:
                return res
        record = {}
        for i, tac_seq in tac_seqs:
            tot = 0
            # print(len(d_is))
            for name, data in d_is:
                # print(i, name, len(tac_seq))
                if self.solve_cache[i].get(name) is None:
                    tot += self.TIMEOUT_COST
                else:
                    # print(len(tac_seq), tac_seq)
                    # print(self.solve_cache[i][name])
                    tot += self.solve_cache[i][name][-len(tac_seq)]
            # print(tot, str(tac_seq[0]))
            tot /= len(d_is)
            if res[0] > tot:
                res = (tot, tac_seq[0])
            # if record.get(str(tac_seq[0])) is None:
            #     record[str(tac_seq[0])] = tot
            # elif record[str(tac_seq[0])] > tot:
            #     record[str(tac_seq[0])] = tot
        # print("min tac printing================")
        # for a, b in record.items():
        #     print(a, b)
        # print("================================")
        return res

    def save_cache(self, outfile, content):
        if outfile is None or content is None:
            return
        with open(outfile, 'a') as f:
            f.write('\n')
            f.write(str(content))
            f.close()
        # with open("_"+outfile, 'w') as f:
        #     f.write(str(self.probe_dict))

    def load_cache(self, outfile):
        if outfile is None:
            return
        if not os.path.exists(outfile):
            return
        if self.solve_cache is None:
            self.solve_cache = {}

        now_i = 0
        with open(outfile, 'r+')  as f:
            lines = f.readlines()
            for line in lines:
                if now_i % 10 == 0:
                    print("load {}th data".format(now_i))
                if len(line) > 1:
                    self.solve_cache[now_i] = eval(line)
                else:
                    print("skip")
                    continue
                # print(self.solve_cache[now_i].keys())
                now_i += 1
        print("load total {} datas".format(now_i))

        # with open("_"+outfile, 'r+') as f:
        #     self.probe_dict = eval(f.readline())

        print("load success")
        # with open(outfile, 'a+') as f:
        #     f.truncate(0)

    def init_solve_cache(self, datas, tac_seqs, collect_probes=False):
        print('=========start to init solve cache:')
        n_data = []
        if self.solve_cache is None:
            self.solve_cache = {}

        for data_i, data in enumerate(datas):
            n_data.append((data, z3.parse_smt2_file(data)))

        for tac_i, tac_seq in tac_seqs:
            if self.solve_cache.get(tac_i) is None:
                self.solve_cache[tac_i] = {}
            else:
                # assert len(tac_seq)+1 == len(list(self.solve_cache[tac_i].values())[0]), "{} {}:{}".format(tac_i, str([str(t) for t in tac_seq]), list(self.solve_cache[tac_i].values())[0])
                print("skip {}th tac_seq".format(tac_i))
                continue
            print("===evaluate {}th tac_seq".format(tac_i))

            tmp_cache = {}
            for data_i, data in enumerate(datas):
                if data_i % 50 == 0:
                    print("evaluate {}th formula".format(data_i))
                # print(self.solve_cache[tac_i].keys())
                # print(self.solve_cache[tac_i][data])
                # if self.solve_cache[tac_i].get(data) is not None:
                    # print("use cache")
                    # continue
                formula = z3.parse_smt2_file(data)
                
                rtime = 0
                res_seq = [0.0]

                def append_seq():
                    if collect_probes:
                        g = z3.Goal()
                        g.add(formula)
                        for p in z3.probes():
                            if self.probe_dict.get(p) is None:
                                self.probe_dict[p] = set()
                            self.probe_dict[p].add(z3.Probe(p)(g))
                    res_seq.append(rtime)

                    #     if self.next_cache[str(formula)].get(str(tac)) is not None:
                    #         rtime, n_formula = self.r_cache[str(formula)][str(tac)], self.next_cache[str(formula)][str(tac)]
                    #         formula = n_formula
                    #         append_seq()
                    #         continue
                for tac in tac_seq:
                    try:
                        res, rtime, n_formula = self.solver.solve_goal(formula, tac.tactic, use_rlimit=True)
                    except z3.z3types.Z3Exception:
                        res, rtime, n_formula = 'unknown', 50000000, None

                    if n_formula is None:
                        break

                    # if self.r_cache.get(str(formula)) is None:
                    #     self.next_cache[str(formula)] = {}
                    #     self.r_cache[str(formula)] = {}
                    # self.r_cache[str(formula)][str(tac)] = rtime
                    # self.next_cache[str(formula)][str(tac)] = n_formula
                    formula = n_formula

                    append_seq()

                if res != 'unknown':
                    for i in range(len(res_seq)):
                        if i > 0:
                            res_seq[len(res_seq)-i-1] += res_seq[len(res_seq)-i]
                    tmp_cache[data] = res_seq
                    #print(res_seq)
            self.save_cache(self.cache_path, tmp_cache)

        print('=========solve cache init finished')
        return n_data


class QuickCombiner(Combiner):
    def __init__(self, solver, cache_path=None):
        super().__init__(solver, cache_path)
        self.TIMEOUT_COST = 5e10

    def init_solve_cache(self, datas, tac_seqs, collect_probes=False):
        print('=========start to init quick solve cache:')
        n_data = []
        if self.solve_cache is None:
            self.solve_cache = {}

        for data_i, data in enumerate(datas):
            n_data.append((data, z3.parse_smt2_file(data)))

        for tac_i, tac_seq in tac_seqs:
            if self.solve_cache.get(tac_i) is None:
                self.solve_cache[tac_i] = {}
            else:
                continue

            print("===evaluate {}th tac_seq".format(tac_i))

            tmp_cache = {}
            for data_i, formulas in enumerate(n_data):
                if data_i % 50 == 0:
                    print("evaluate {}th formula".format(data_i))

                data, formula = formulas
                tac = objects.AndThen(*tac_seq) if len(tac_seq) > 1 else tac_seq[0]

                try:
                    res, rtime, _ = self.solver.solve_goal(formula, tac.tactic, use_rlimit=True)
                except z3.z3types.Z3Exception:
                    res, rtime = 'unknown', 5e7

                if str(res) != 'unknown':
                    tmp_cache[data] = rtime
                    
            self.save_cache(self.cache_path, tmp_cache)

        print('=========solve cache init finished')
        return n_data

    def find_min_tac(self, d_is, tac_seqs):
        res = (1e20, objects.Tactic('skip'))
        if len(tac_seqs) == 0:
            return res
        if len(d_is) == 0:
            tac_seq = tac_seqs[0][1]
            if len(tac_seq) > 0:
                return (1e19, tac_seqs[0][1][0])
            else:
                return res
        record = {}
        for i, tac_seq in tac_seqs:
            tot = 0
            for name, data in d_is:
                if self.solve_cache[i].get(name) is None:
                    tot += self.TIMEOUT_COST
                else:
                    tot += self.solve_cache[i][name]
            tot /= len(d_is)
            if res[0] > tot:
                res = (tot, tac_seq[0])
        return res



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', type=str, default='experiments/configs/normal_config.json')
    parser.add_argument('--tactics', type=str, default='tuner_tactic.txt')
    parser.add_argument('--train_data', type=str, default='../experiments/data/coreutils/train')
    parser.add_argument('--cache_path', type=str, default='solve_cache.cache')
    parser.add_argument('--old_type', type=bool, default=False)
    parser.add_argument('--valid_data', type=str, default='None')
    args = parser.parse_args()

    data = []
    for root, directories, filenames in os.walk(args.train_data):
        for file in filenames:
            if file.endswith('.smt2'):
                data.append(os.path.join(root, file))
    if args.valid_data != 'None':
        for root, directories, filenames in os.walk(args.valid_data):
            for file in filenames:
                if file.endswith('.smt2'):
                    data.append(os.path.join(root, file))
        

    # data = data[:20]
    import random
    from agent import parse_combine_tactic
    # data = random.sample(data, 2000)
    print(len(data))

    tac_seqs = []
    with open(args.tactics, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # print("parsing")
            adt = parse_combine_tactic(line)
            print(str(adt))

            tac_seqs.append(adt.v if isinstance(adt, objects.AndThen) else [adt])
        f.close()

    tokenizer = GoalTokenizer()
    enumrator = StrategyEnumerator(**json.load(open(args.configuration, 'r'))['tactics_config'])
    
    if args.old_type:
        cb = Combiner(SMTSolver(tokenizer, enumrator), args.cache_path)
    else:
        cb = QuickCombiner(SMTSolver(tokenizer, enumrator), args.cache_path)

    result = cb.gen_strategy(data, tac_seqs)
    print(str(result))
    print(str(result.to_smt2()))


if __name__ == '__main__':
    main()
