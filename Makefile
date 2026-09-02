.PHONY: audit-data watch-data prepare-essays prepare-goemotions prepare-bfi2-bessi train-goemotions train-model1-piastra train-model1-prompt-ensemble-prepare train-model1-prompt-ensemble-submit train-model1-prompt-ensemble-retry train-model1-prompt-ensemble-resubmit train-model1-prompt-ensemble-status train-model1-prompt-ensemble-collect train-model1-prompt-ensemble-direct train-model1-luna-specialists train-model1-luna-joint train-model1-local-pilot train-model1-local-validation train-model1-local-train train-model1-local-stacker train-model1-ablation train-model1-embeddings train-model2-benchmark train-model2-improved evaluate-model2-selected test

audit-data:
	.venv/bin/python src/data/audit_raw_datasets.py

watch-data:
	.venv/bin/python src/data/watch_raw_datasets.py

prepare-essays:
	python3 src/data/prepare_essays.py

prepare-goemotions:
	python3 src/data/prepare_goemotions.py

prepare-bfi2-bessi:
	.venv/bin/python src/data/prepare_bfi2_bessi.py

train-goemotions:
	.venv/bin/python src/models/train_goemotions.py

train-model2-benchmark:
	.venv/bin/python src/models/train_model2_benchmark.py

train-model2-improved:
	.venv/bin/python src/models/train_model2_improved.py

evaluate-model2-selected:
	.venv/bin/python src/models/evaluate_model2_selected.py --confirm-final-test

train-model1-piastra:
	.venv/bin/python src/models/score_model1_piastra.py --env-file .env --confirm-remote-inference

train-model1-prompt-ensemble-prepare:
	.venv/bin/python src/models/score_model1_prompt_ensemble_batch.py prepare

train-model1-prompt-ensemble-submit:
	.venv/bin/python src/models/score_model1_prompt_ensemble_batch.py submit --confirm-remote-inference

train-model1-prompt-ensemble-retry:
	.venv/bin/python src/models/score_model1_prompt_ensemble_batch.py retry --confirm-remote-inference

train-model1-prompt-ensemble-resubmit:
	.venv/bin/python src/models/score_model1_prompt_ensemble_batch.py resubmit --confirm-remote-inference

train-model1-prompt-ensemble-status:
	.venv/bin/python src/models/score_model1_prompt_ensemble_batch.py status

train-model1-prompt-ensemble-collect:
	.venv/bin/python src/models/score_model1_prompt_ensemble_batch.py collect

train-model1-prompt-ensemble-direct:
	.venv/bin/python -u src/models/score_model1_prompt_ensemble_batch.py direct --confirm-remote-inference --max-estimated-usd 2.00

train-model1-luna-specialists:
	.venv/bin/python -u src/models/score_model1_luna_specialists.py --confirm-remote-inference --max-estimated-usd 1.50

train-model1-luna-joint:
	.venv/bin/python -u src/models/score_model1_luna_joint.py --confirm-remote-inference --max-estimated-usd 0.20

train-model1-local-pilot:
	.venv/bin/python src/models/score_model1_local_llm.py --limit 20 --predictions outputs/model-evaluation/model1_local_llm_pilot_scores.csv --report reports/model-evaluation/model1-local-llm-pilot-report.md --metadata data/metadata/model1-local-llm-pilot.json --partial-predictions outputs/model-evaluation/model1_local_llm_pilot_scores.partial.csv

train-model1-local-validation:
	.venv/bin/python src/models/score_model1_local_llm.py

train-model1-local-train:
	.venv/bin/python src/models/score_model1_local_llm.py --essays data/processed/model1/essays_train.csv --split train --predictions outputs/model-evaluation/model1_local_llm_train_scores.csv --report reports/model-evaluation/model1-local-llm-train-scores-report.md --metadata data/metadata/model1-local-llm-train-scores.json --partial-predictions outputs/model-evaluation/model1_local_llm_train_scores.partial.csv

train-model1-local-stacker:
	.venv/bin/python src/models/train_model1_local_llm_stacker.py

train-model1-ablation:
	.venv/bin/python src/models/train_model1_emotion_ablation.py

train-model1-embeddings:
	.venv/bin/python src/models/train_model1_embedding_ablation.py

test:
	.venv/bin/python -m unittest discover -s tests
