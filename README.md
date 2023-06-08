# smt_testing

本程序主要用于z3背景下的SMT求解加速使用，内含部分对FastSMT项目文件的引用，相关内容可参见：

https://github.com/eth-sri/fastsmt/

环境配置可参考如下信息

```shell
virtual -p /usr/bin/python3.6 py3env
source py3env/bin/activate
pip install z3-solver
pip install timeout-decorator
pip install torch torchvision torchaudio

```

实验数据集需要由使用方提供，程序运行方式可见examples文件夹相关示例，一次完整的运行流程为：

```shell
mkdir -p cache/model
mkdir tmp
./examples/gen_tactic_seqs.sh
./examples/filter_tactic_seqs.sh
./examples/quick_tuner.sh
./examples/strategy_gen.sh
```

在得到.tac文件后(使用SMT-LIB描述的Strategy)可通过如下命令与Z3求解器进行效率对比：

```shell
python3 scripts/validate.py \
    --strategy_file xxx.tac \
    --batch_size 4          \
    --max_timeout 5         \
    --benchmark_dir experiments/data/xx/all \
    | tee xxx.log
```

各步骤应获取的中间及对应格式可于experiments/results/core_exp1中查看
