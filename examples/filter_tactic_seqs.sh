python3 -u agent.py \
	--mode collect_tactic \
	--tactics core_sample_gen.tacs \
	--train_data experiments/data/core/train \
	| tee core_sample_gen.tac
