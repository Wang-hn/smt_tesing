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

import json
import logging
import random
import re
import time
import z3
import os
import argparse
import threading
import subprocess
import shlex

import timeout_decorator

import torch
import torch.nn as nn

from utils.strategy import StrategyEnumerator
from language import objects


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
    'int', 'bool', 'bitVec',
]

ALL_TOKENS = ["UNK"] + ST_TOKENS + BV_THEORY


class Z3Runner(threading.Thread):
    def __init__(self, smt_file, timeout, strategy=None, id=1):
        threading.Thread.__init__(self)
        self.smt_file = smt_file
        self.timeout = timeout
        self.strategy = strategy
        RND = 'tmp'

        if self.strategy is not None:
            self.tmp_file = open('tmp/tmp_valid_{}_{}.smt2'.format(RND, id), 'w')
            with open(self.smt_file, 'r') as f:
                for line in f:
                    new_line = line
                    if 'check-sat' in line:
                        new_line = '(check-sat-using %s)\n' % strategy
                    self.tmp_file.write(new_line)
            self.tmp_file.close()
            self.new_file_name = 'tmp/tmp_valid_{}_{}.smt2'.format(RND, id)
        else:
            self.new_file_name = self.smt_file

    def run(self):
        self.time_before = time.time()
        z3_cmd = 'z3 -smt2 %s -st' % self.new_file_name
        
        self.p = subprocess.Popen(shlex.split(z3_cmd), stdout=subprocess.PIPE)
        self.p.wait()
        self.time_after = time.time()

    def collect(self):
        if self.is_alive():
            try:
                self.p.terminate()
                self.join()
            except OSError:
                pass
            return None, None, None

        out, err = self.p.communicate()

        lines = out[:-1].decode("utf-8").split('\n')
        res = lines[0]

        rlimit = None
        for line in lines:
            if 'rlimit' in line:
                tokens = line.split(' ')
                for token in tokens:
                    if token.isdigit():
                        rlimit = int(token)

        if res == 'unknown':
            res = None

        return res, rlimit, self.time_after-self.time_before

class GoalTokenizer(object):
    def __init__(self):
        self.token_idx = {}
        for token_i, token in enumerate(ALL_TOKENS):
            self.token_idx[token] = token_i

    def bow(self, txt):
        if type(txt) is not str:
            txt = str(txt)

        txt = re.sub(r'[()\n]', ' ', txt)
        txt = re.sub(r'[\[\],]', ' ', txt)
        txt = re.sub(r"\[|]+", ' ', txt)

        tokens = txt.split(' ')

        ret = [0 for _ in ALL_TOKENS]
        for token in tokens:
            token = token.lower()
            if token not in self.token_idx:
                ret[0] += 1
            else:
                ret[self.token_idx[token]] += 1
        return ret


class SampleBuffer:
    def __init__(self, memory_size, batch_size):
        self.MEM_SIZE = memory_size
        self.BATCH_SIZE = batch_size

        self.iter = 0

        self.buf = []
        self.main_buf = []
        self.bf_rate = 0.4

    def add_sample(self, s, a, r, done, s_):
        if not done:
            self.buf.append([s, a, r, done, s_])
        else:
            self.main_buf.append([s, a, r, done, s_])

        if len(self.buf) > self.MEM_SIZE:
            self.buf.pop(0)
        if len(self.main_buf) > self.MEM_SIZE:
            self.main_buf.pop(0)

    def sample(self):
        if len(self.buf) + len(self.main_buf) <= 1:
            return None, None, None, None, None
        main_num = int(self.BATCH_SIZE * 0.1)
        ano_num = self.BATCH_SIZE - main_num

        if len(self.main_buf) > 0:
            ft_main = random.sample(self.main_buf, min(main_num, len(self.main_buf)))
        else:
            ft_main = []

        ano_num += main_num - len(ft_main)

        ft = random.sample(self.buf, min(ano_num, len(self.buf)))
        ft = ft + ft_main
        return tuple(zip(*ft))


class DQN(nn.Module):
    def __init__(self, num_probes, num_tactic):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_probes, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, num_tactic),
            nn.ReLU()
        )
        self.loss = nn.SmoothL1Loss()
        # for m in self.net.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight)
        #          # nn.init.kaiming_normal_(m.weight)
        #         print(m.weight)

    def forward(self, input_data):
        return self.net(input_data)

    def do_train(self, input_data, act, target):
        self.train(True)
        output = self.forward(input_data)

        a_q_values = torch.gather(input=output, index=act, dim=1)

        res_loss = self.loss(a_q_values, target)
        print(res_loss)
        res_loss.backward()

    def predict(self, input_data):
        self.eval()
        output = self.forward(input_data)
        return output


class SMTSolver:
    def __init__(self, tokenizer, enumerator):
        self.enumerator = enumerator
        self.tokenizer = tokenizer
        pass

    @timeout_decorator.timeout(5, use_signals=False)
    def try_to_solve_5(self, solver, formula):
        solver.add(formula)
        solver.check()

    @timeout_decorator.timeout(10, use_signals=False)
    def try_to_solve_10(self, solver, formula):
        solver.add(formula)
        solver.check()

    def solve_without_timeout(self, formula, tactic):
        if type(tactic) is str:
            tactic = z3.Tactic(tactic)
        s = tactic.solver()

        s.check()
        r_before = self.get_rlimit(s)
        s.add(formula)
        res = s.check()
        r_after = self.get_rlimit(s)

        # g = z3.Goal()
        # g.add(formula)

        # if isinstance(tactic, z3.Tactic) : print("true")
        return str(res), r_after - r_before, s.assertions()

    def solve_with_tactic_seq(self, formula, tactics, collect_probs=False):
        if collect_probs:
            ps = [(i, z3.Probe(i)) for i in z3.probes()]
            pm = {i: set() for i in z3.probes()}
        else:
            ps, pm = None, None
            if not collect_probs:
                tac = objects.AndThen(*tactics) if len(tactics)>1 else tactics[0]
                tac = z3.TryFor(tac.tactic, 5000)
                s = tac.solver()
                s.check()
                r_before = self.get_rlimit(s)
                s.add(formula)
                res = s.check()
                r_after = self.get_rlimit(s)
                return str(res), r_after-r_before, s.assertions(), pm

        def feather_probs(formula):
            if not collect_probs:
                return
            g = z3.Goal()
            g.add(formula)
            for name, pb in ps:
                pm[name].add(pb[formula])
            return

        tactics.insert(0, z3.Tactic('skip'))
        tot_rlimit = 0
        res = 'unknown'

        for tac in tactics:
            if not isinstance(tac, z3.Tactic):
                tac = tac.tactic
            tac = z3.TryFor(tac, 5000)
            
            res, rlimit, formula = self.solve_without_timeout(formula, tac)
            tot_rlimit += rlimit
            feather_probs(formula)

        return str(res), tot_rlimit, formula, pm

    def solve(self, formula, tactic):
        if type(tactic) is str:
            tactic = z3.Tactic(tactic)
        tactic = z3.TryFor(tactic, 5000)
        s = tactic.solver()
        '''
        try:
            # print('try begin')
            self.try_to_solve_5(s, formula)
            # print('try success')
        except timeout_decorator.timeout_decorator.TimeoutError as e:
            # print('timeout')
            return None, 'unknown', 1000000, 100000, None, formula
        '''

        s.check()
        s.add(formula)

        r_before = self.get_rlimit(s)
        t_before = time.time()

        res = s.check()

        t_after = time.time()
        r_after = self.get_rlimit(s)

        res = str(res)

        rlimit = r_after - r_before
        rtime = t_after - t_before

        g = z3.Goal()
        g.add(formula)
        s1 = self.get_probs(g) + self.tokenizer.bow(formula.sexpr())
        g = z3.Goal()
        g.add(s.assertions())

        s_ = self.get_probs(g) + self.tokenizer.bow(formula.sexpr())

        return s1, res, rlimit, rtime * 1000, s_, s.assertions()

    @staticmethod
    def get_probs(goal):
        probs = [z3.Probe(p) for p in z3.probes()]
        ng = goal.as_expr()
        probs = [p(ng) for p in probs]
        return probs

    def solve_dataset(self, formulas, tactic, timeout=5):
        tactic = z3.TryFor(tactic, timeout * 1000)
        solver = tactic.solver()

        unsolved = 0
        tot_rlimit = 0
        for formula in formulas:
            solver.check()
            r_before = self.get_rlimit(solver)
            solver.from_file(formula)
            res = solver.check()
            if res == z3.unknown:
                unsolved += 1
            else:
                r_after = self.get_rlimit(solver)
                tot_rlimit += r_after - r_before
            solver.reset()

        return unsolved, tot_rlimit

    @staticmethod
    def get_rlimit(s):
        stats = s.statistics()
        for i in range(len(stats)):
            if stats[i][0] == 'rlimit count':
                return stats[i][1]
        return 0

    @timeout_decorator.timeout(30, use_signals=False)
    def solve_by_z3(self, formula):
        # formula = z3.parse_smt2_file(smt_instance)
        s = z3.Solver()
        s.check()
        before = self.get_rlimit(s)
        s.add(formula)
        t_before = time.time()
        res = s.check()
        t_after = time.time()
        print("z3: ", res, self.get_rlimit(s) - before, t_after - t_before)

    @timeout_decorator.timeout(30, use_signals=False)
    def solve_by_tactic(self, formula, tactic):
        if type(tactic) is not z3.Tactic:
            tactic = tactic.tactic
        s = tactic.solver()
        s.check()
        r_before = self.get_rlimit(s)
        s.add(formula)
        t_before = time.time()
        res = s.check()
        t_after = time.time()
        print("predict: ", res, self.get_rlimit(s) - r_before, t_after - t_before)

    def solve_goal(self, formula, tac, use_rlimit=False, timeout=5):
        g = z3.Goal()
        g.add(formula)
        if isinstance(tac, str):
            tac = z3.Tactic(tac)
        if not isinstance(tac, z3.Tactic):
            tac = tac.tactic
        tac = z3.TryFor(tac, timeout * 1000)
        gs = tac.solver()

        if use_rlimit:
            gs.add(formula)
            r_before = self.get_rlimit(gs)
            res = gs.check()
            r_after = self.get_rlimit(gs)
            return res, r_after-r_before, gs.assertions()

        t_before = time.time()
        g = tac(g)
        t_after = time.time()
        s = z3.Tactic('skip').solver()
        s.add(g[0].as_expr())
        return str(s.check()), t_after - t_before, g[0].as_expr()


class Agent:
    def __init__(self, config, episode_cnt, step_cnt, rand_tactic_num, exp_name, out_file):
        self.enumerator = StrategyEnumerator(**config["tactics_config"])
        self.tokenizer = GoalTokenizer()

        self.online_net = DQN(118, len(self.enumerator.all_tactics))
        self.target_net = DQN(118, len(self.enumerator.all_tactics))

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=0.001)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, gamma=0.9)

        self.buf = SampleBuffer(2000, 100)
        self.solver = SMTSolver(self.tokenizer, self.enumerator)

        self.episode_cnt = episode_cnt
        self.rand_num = rand_tactic_num
        self.logging = logging.getLogger('DQNAgent')
        self.gamma = 0.5
        self.step_cnt = step_cnt
        self.trans_cnt = 200

        self.exp_name = exp_name
        self.best_strategy = {}
        self.r_denominator = 1.0
        self.out_file = out_file

    def construct_denominator(self, smt_instances, use_rlimit=False):
        # smt_instances = random.sample(smt_instances, int(len(smt_instances) / 2))
        tot_r = 0
        print("=================start to initial denominator:")
        # r_list = []
        for smt_i, smt_instance in enumerate(smt_instances):
            if smt_i % 5 == 0:
                print("evaluate {}th formula".format(smt_i))
            formula = z3.parse_smt2_file(smt_instance)
            try:
                s = z3.Solver()
                self.solver.try_to_solve_5(s, formula)
                s = z3.Solver()
                if use_rlimit:
                    s.check()
                    s.add(formula)
                    r_before = self.solver.get_rlimit(s)
                else:
                    s.add(formula)
                    r_before = time.time()
                
                s.check()

                if use_rlimit:
                    r_after = self.solver.get_rlimit(s)
                else:
                    r_after = time.time()
                tot_r += r_after - r_before
            except timeout_decorator.timeout_decorator.TimeoutError:
                tot_r += 6
        avg_r = tot_r * 1000 / len(smt_instances)
        self.r_denominator = avg_r / 10
        print("======================construct result:", avg_r / 10)

    def train(self, smt_instances, record_best=False):
        self.construct_denominator(smt_instances)
        self.target_net.load_state_dict(self.online_net.state_dict())
        if record_best:
            self.best_strategy = {instance: (-1e12, []) for instance in smt_instances}
        turn_cnt = 0
        for ep_i in range(self.episode_cnt):
            print("================ep:%d====================" % ep_i)
            for ins_i in range(len(smt_instances)):
                for repeat_i in range(3 if ep_i < self.episode_cnt-1 else 1):
                    instance = smt_instances[ins_i]
                    formula = z3.parse_smt2_file(instance)
                    s, _, _, _, s_, formula = self.solver.solve(formula, 'skip')

                    episode_reward = 0
                    tac_memory = {tac: 0 for tac in self.enumerator.all_tactics}
                    s = s + list(tac_memory.values())
                    lst_tactic = []
                    tac_seq = []
                    loop = False

                    print('===try to solve {}th formula: '.format(ins_i), smt_instances[ins_i])
                    for st_i in range(self.step_cnt if ep_i > 0 else self.step_cnt * 2):
                        turn_cnt += 1

                        if len(lst_tactic) == len(self.enumerator.all_tactics):
                            break

                        tensor_s = torch.as_tensor(s).unsqueeze(0)
                        wait = self.online_net.predict(tensor_s)[0]
                        # print(wait)

                        if ep_i < self.rand_num:
                            act = random.choice(self.enumerator.all_tactics)
                            while loop and act in lst_tactic:
                                act = random.choice(self.enumerator.all_tactics)
                        elif ep_i < self.episode_cnt-1:
                            wait = wait.sub(torch.min(wait)).add_(0.5)
                            torch.manual_seed(int(random.random() * 100000))
                            wait = nn.functional.normalize(wait, dim=0)
                            while wait[torch.argmin(wait)] < 0:
                                wait[torch.argmin(wait)] = 0.0
                            act = self.find_tactic(torch.multinomial(wait, 1)[0])
                            while loop and act in lst_tactic:
                                wait[torch.argmax(wait)] = -10000
                                act = self.find_tactic(int(torch.argmax(wait)))
                        else:
                            act = self.find_tactic(int(torch.argmax(wait)))

                        tac = z3.Tactic(act)

                        ind = self.enumerator.all_tactics.index(act)
                        ori_s = s

                        try:
                            s, res, rlimit, rtime, s_, formula = self.solver.solve(formula, tac)
                        except z3.z3types.Z3Exception:
                            s, res, rlimit, rtime, s_ = None, 'unknown', 1000000, 5000, None

                        if s is None or rlimit == 0:
                            s, s_ = ori_s, ori_s
                        else:
                            s = s + list(tac_memory.values())

                        if res == 'unknown' and s == s_:
                            loop = True
                            lst_tactic.append(act)
                            self.buf.add_sample(s, ind, -10, True if res != 'unknown' else False, s_)
                            continue
                        else:
                            episode_reward -= rtime
                            bias = 20 - tac_memory[act] * 10
                            bias = max(bias, -10)
                            tac_seq.append(act)
                            tac_memory[act] += 1
                            s_ = s_ + list(tac_memory.values())
                            print(str(act))

                        loop = False
                        lst_tactic.clear()

                        bias += 0 if res == 'unknown' else 20
                        # print(rtime, bias, self.r_denominator)
                        if len(s) == len(s_):
                            self.buf.add_sample(s, ind, bias - rtime / self.r_denominator, True if res != 'unknown' else False, s_)

                        self.logging.info('%d: %d' % (ep_i, episode_reward))

                        batch_s, batch_a, batch_r, batch_res, batch_s_ = self.buf.sample()
                        # print(torch.as_tensor(batch_s_))
                        # print(batch_r)
                        if batch_s is not None:
                            target_q_values = self.target_net(torch.as_tensor(batch_s_))
                            target_max_values = target_q_values.max(dim=1, keepdim=True)[0]
                            # print(target_max_values)
                            # print(batch_r)

                            targets = torch.transpose(torch.as_tensor([batch_r]), 1, 0) + self.gamma * target_max_values
                            # print(targets)
                            

                            self.optimizer.zero_grad()
                            self.online_net.do_train(torch.as_tensor(batch_s),
                                                     torch.transpose(torch.as_tensor([batch_a]), 1, 0), targets)
                            self.optimizer.step()
                        # print("step done")
                        if res != 'unknown':
                            if record_best and self.best_strategy[instance][0] < episode_reward:
                                self.best_strategy[instance] = (episode_reward, tac_seq)
                            print('formula solved!')
                            break

                    if turn_cnt % self.trans_cnt == 0:
                        self.target_net.load_state_dict(self.online_net.state_dict())
                        print('ep:%d change model success; reward: %d' % (ep_i, episode_reward))
                    else:
                        print('ep:%d is done; reward: %d' % (ep_i, episode_reward))

            print("================ep:%d====================" % ep_i)
            self.output_best_strategy()
            torch.save(self.online_net.state_dict(), 'cache/model/{}_{}_5.pth'.format(self.exp_name, ep_i))


    @timeout_decorator.timeout(30, use_signals=False)
    def predict(self, formula, random_select=False):
        print('try to predict:')

        tot_rlimit = 0

        tactic_seq = []
        tac_memory = {tac: 0 for tac in self.enumerator.all_tactics}
        t_before = time.time()

        done = False
        print("begin")

        for i in range(30):
            g = z3.Goal()
            g.add(formula)
            s = self.solver.get_probs(g) + self.tokenizer.bow(formula.sexpr()) + list(tac_memory.values())
            a = self.online_net.predict(torch.as_tensor(s).unsqueeze(0))
            print(a)
            if random_select:
                a = torch.where(torch.lt(a, 0), torch.zeros_like(a), a)
                a = a[0].add(0.5)
                torch.manual_seed(int(random.random() * 100000))
                wait = torch.multinomial(a, 18, replacement=True)
                act = self.find_tactic(wait[0])
            else:
                act = self.find_tactic(int(torch.argmax(a)))
            if act == 'bit-blast' and 'simplify' not in tactic_seq:
                act = 'simplify'
            print("try to use", act)

            res, rlimit, n_formula = self.solver.solve_without_timeout(formula, act)
            if s is None:
                print('failed')
                return

            print(act, rlimit)
            tot_rlimit += rlimit
            tac_memory[act] += 1
            if random_select and str(n_formula) == str(formula) and res == 'unknown':
                tot_rlimit -= rlimit
            else:
                tactic_seq.append(act)

            formula = n_formula
            if res != 'unknown':
                done = True
                break
        print('success' if done else "failed", tot_rlimit, time.time() - t_before)
        print(tactic_seq)

    def extract_tactics(self, eva_tuples):
        for instance, r, tac in eva_tuples:
            formula = z3.parse_smt2_file(instance)
            s1 = z3.Solver()
            s2 = z3.TryFor(tac.tactic, 5000).solver()

            def get_check_time(s):
                s.check()
                rlimit_before = self.solver.get_rlimit(s)
                s.add(formula)
                rtime_before = time.time()
                res = s.check()
                rtime_after = time.time()
                rlimit_after = self.solver.get_rlimit(s)
                return str(res), rlimit_after-rlimit_before, rtime_after - rtime_before

            res, r2, t2 = get_check_time(s2)
            try:
                self.solver.try_to_solve_5(s1, formula)
                s1 = z3.Solver()
                _, r1, t1 = get_check_time(s1)
            except timeout_decorator.timeout_decorator.TimeoutError:
                r1, t1 = r2+1, r2+1

            if r2 > r1 and t2 > t1 and res != 'unknown':
                print('best', str(tac))
            elif r2 > r1 and res != 'unknown':
                print('rlimit', str(tac))
            elif t2 > t1 and res != 'unknown':
                print('time', str(tac))
            else:
                print('None!')
            if res == 'timeout':
                print('timeout!')

    def extract_tactics_with_runner(self, eva_tuples):
        idx = 1
        task1 = []
        task2 = []
        for instance, r, tac in eva_tuples:
            runner1 = Z3Runner(instance, 20, tac.to_smt2(), idx)
            runner2 = Z3Runner(instance, 20)
            idx = idx+1

            runner1.start()
            runner1.join(20)
            res1, rlimit1, time1 = runner1.collect()

            runner2.start()
            runner2.join(20)
            res2, rlimit2, time2 = runner2.collect()

            if res1 is not None:
                if res2 is None or (time1 < time2 and rlimit1 < rlimit2):
                    print("best", str(tac))
                elif time1 < time2:
                    print("time", str(tac))
                elif rlimit1 < rlimit2:
                    print("rlimit", str(tac))
                else:
                    self.logging.info("z3 better")
            elif res2 is None:
                self.logging.info("timeout")
            else:
                self.logging.info("only z3")

    def one_hot_action(self, act):
        return [1 if act == i else 0 for i in range(len(self.enumerator.all_tactics))]

    def find_tactic(self, index):
        return self.enumerator.all_tactics[index]

    def output_best_strategy(self):
        print("===============start to print strategy")
        if self.out_file is not None:
            with open(self.out_file, 'w') as f:
                for tac_seq in self.best_strategy.values():
                    f.write(str(tac_seq)+"\n")
        for tac_seq in self.best_strategy.values():
            print(tac_seq)
        print("===============over")


def parse_tactic(line):
    if line is None or line == '':
        return None
    if isinstance(line, str):
        line = eval(line)
    tac_seq = []
    for tac in line:
        if tac == '' or tac is None:
            break
        if type(tac) is str:
            tac_seq.append(objects.Tactic(tac))
            continue
        tactic, params = tac
        tac_seq.append(objects.With(tactic, params))
    return objects.AndThen(*tac_seq) if len(tac_seq) > 1 else tac_seq[0]

import combiner
def parse_combine_tactic(line):
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
        while line.find(';') > 0:
            pos1 = line.find(';')
            pos2 = line.find('=')
            dic[line[:pos2]] = eval(line[pos2 + 1: pos1])
            line = line[pos1+1:]
        pos1 = line.find('=')
        dic[line[:pos1]] = eval(line[pos1+1:line.find(')')])
        return objects.With(tac1, dic)
    if line.startswith('Cond'):
        balance = 0
        pos1 = line.find(',')
        pos2 = line.find('>')
        pb1 = parse_combine_tactic(line[5: pos2 - 1])
        num1 = float(line[pos2 + 1: pos1])
        pb1 = combiner.ProbeCond(pb1, num1)

        line = line[pos1:]

        tac_v = []

        for i in range(len(line)):
            if line[i] == ' ' and balance == 0:
                pos1 = i + 1
            if line[i] == '(':
                balance += 1
            if line[i] == ')':
                balance -= 1
                if balance == 0:
                    tac_v.append(parse_combine_tactic(line[pos1: i + 1]))
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
                    tac_v.append(parse_combine_tactic(line[pos1: i + 1]))

        return objects.AndThen(*tac_v)
    if line.startswith('['):
        pos1, pos2 = 0, 0
        tac_v = []
        while True:
            pos1 = line.find('\'', pos2+1)
            if pos1 == -1:
                break
            pos2 = line.find('\'', pos1+1)
            tac_v.append(parse_combine_tactic(line[pos1+1:pos2]))
        return objects.AndThen(*tac_v) if len(tac_v) > 1 else tac_v[0]


    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='None')
    parser.add_argument('--tactics', type=str, default='tuner_tactic.txt')
    parser.add_argument('--train_data', type=str, default='../experiments/data/coreutils/train')
    parser.add_argument('--test_data', type=str, default='../experiments/data/core/test')
    parser.add_argument('--configuration', type=str, default='experiments/configs/normal_config.json')
    parser.add_argument('--random_select', type=bool, default=False)
    parser.add_argument('--exp_name', type=str, default='coreutils_exp5')
    parser.add_argument('--collect_best', type=bool, default=True)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--episode_cnt', type=int, default=5)
    parser.add_argument('--apply_cnt', type=int, default=10)
    parser.add_argument('--random_ep_cnt', type=int, default=1)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    agent = Agent(json.load(open(args.configuration, 'r')), args.episode_cnt, args.apply_cnt, args.random_ep_cnt, args.exp_name, out_file=args.out_file)
    # agent.output_best_strategy()

    if args.mode == 'train':
        data = []
        for root, directories, filenames in os.walk(args.train_data):
            for file in filenames:
                if file.endswith('.smt2'):
                    data.append(os.path.join(root, file))

        print(len(data))
        if args.model is None:
            agent.online_net.load_state_dict(torch.load(args.model))

        agent.train(data, args.collect_best)
        if args.collect_best:
            agent.output_best_strategy()

    elif args.mode == 'test':
        # print(torch.load(args.model))
        agent.online_net.load_state_dict(torch.load(args.model), strict=True)
        agent.online_net.eval()

        for root, directories, filenames in os.walk(args.test_data):
            for file in filenames:
                if file.endswith('smt2'):
                    formula = z3.parse_smt2_file(os.path.join(root, file))
                    _, _, formula = agent.solver.solve_without_timeout(formula, z3.Tactic('skip'))
                    print("begin to evaluate", str(file))
                    try:
                        agent.solver.solve_by_z3(formula)
                    except timeout_decorator.timeout_decorator.TimeoutError:
                        print("z3 timeout")
                    try:
                        agent.predict(formula, args.random_select)
                    except timeout_decorator.timeout_decorator.TimeoutError:
                        print("predict timeout")

    elif args.mode == 'tactic':
        with open(args.tactics, 'r') as f:
            lines = f.readlines()

            for line in lines:
                tactic = parse_tactic(line)
                if tactic is None:
                    continue
                print("=======begin to evaluate tactic sequence: ", str(tactic))

                for root, directories, filenames in os.walk(args.test_data):
                    for file in filenames:
                        if file.endswith('smt2'):
                            formula = z3.parse_smt2_file(os.path.join(root, file))
                            _, _, formula = agent.solver.solve_without_timeout(formula, z3.Tactic('skip'))
                            print("begin to evaluate", str(file))
                            try:
                                agent.solver.solve_by_z3(formula)
                            except timeout_decorator.timeout_decorator.TimeoutError:
                                print("z3 timeout")

                            try:
                                agent.solver.solve_by_tactic(formula, tactic)
                            except timeout_decorator.timeout_decorator.TimeoutError:
                                print("predict timeout")

            f.close()
    elif args.mode == 'combine_tactic':
        with open(args.tactics, 'r') as f:
            line = f.readline()
            tac = parse_combine_tactic(line)
            print("==============begin to evaluate {}".format(str(tac)))
            for root, directories, filenames in os.walk(args.test_data):
                for file in filenames:
                    if file.endswith('smt2'):
                        formula = z3.parse_smt2_file(os.path.join(root, file))
                        _, _, formula = agent.solver.solve_without_timeout(formula, z3.Tactic('skip'))
                        print("begin to evaluate", str(file))
                        try:
                            agent.solver.solve_by_z3(formula)
                        except timeout_decorator.timeout_decorator.TimeoutError:
                            print("z3 timeout")

                        try:
                            agent.solver.solve_by_tactic(formula, tac)
                        except timeout_decorator.timeout_decorator.TimeoutError:
                            print("predict timeout")
        f.close()
    elif args.mode == 'collect_tactic':
        data_dict = {}
        for root, directories, filenames in os.walk(args.train_data):
            for file in filenames:
                if file.endswith('.smt2'):
                    data_dict[os.path.join(root, file)] = 0

        data = list(data_dict.keys())

        with open(args.tactics, 'r') as f:
            line = f.readlines()
            eva_tuple = []
            for i in range(len(data)):
                lst = eval(line[i])
                if lst[1] == []:
                    continue
                reward, tac = lst[0], lst[1]
                eva_tuple.append((data[i], reward, parse_tactic(tac)))
            agent.extract_tactics_with_runner(eva_tuple)
            f.close()
    elif args.mode =='tmp':
        agent.online_net.load_state_dict(torch.load(args.model), strict=True)
        agent.online_net.eval()
        torch.onnx.export(agent.online_net, torch.randn(1, 117), 'tmp.pth')


if __name__ == '__main__':
    main()
