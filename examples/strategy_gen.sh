python3 -u combiner.py \
	--tactics core_sample_tuner.tac \
	--train_data experiments/data/core/train/ \
	--valid_data experiments/data/core/valid/ \
	--cache_path cache/core_sample.cac \
	| tee core_sample_combine.log

tail -1 core_sample_combine.log | tee core_sample.tac
