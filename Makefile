.PHONY: audit-data watch-data prepare-essays prepare-goemotions prepare-bfi2-bessi prepare-bfi2-bessi-facets scenarios generate-scenarios-demo run-integration-fixture run-integration-live-smoke train-goemotions train-model1-piastra train-model1-piastra-test train-model1-prompt-ensemble-prepare train-model1-prompt-ensemble-submit train-model1-prompt-ensemble-retry train-model1-prompt-ensemble-resubmit train-model1-prompt-ensemble-status train-model1-prompt-ensemble-collect train-model1-prompt-ensemble-direct train-model1-luna-specialists train-model1-luna-joint train-model1-luna-retrieval train-model1-luna-retrieval-medium train-model1-luna-retrieval-medium-test train-model1-luna-trait-balanced train-model1-local-pilot train-model1-local-validation train-model1-local-train train-model1-local-stacker train-model1-ablation train-model1-embeddings train-model2-benchmark train-model2-improved train-model2-facets evaluate-model2-selected test

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

prepare-bfi2-bessi-facets:
	.venv/bin/python src/data/prepare_bfi2_bessi_facets.py

generate-scenarios-demo:
	.venv/bin/python -m src.scenarios.generate_scenarios --skills self_management,cooperation,innovation --delivery data/processed/scenarios/demo_scenario_instrument_delivery.csv --audit-bank data/processed/scenarios/demo_scenario_instrument_audit.csv --metadata data/metadata/scenario-instrument-pipeline-demo.json --report reports/scenario-design/demo-scenario-instrument-pipeline.md

scenarios:
	.venv/bin/python -m src.scenarios.generate_scenarios --interactive

run-integration-fixture:
	.venv/bin/python -m src.integration.fixture_pipeline --fixture three --root .

run-integration-live-smoke:
	.venv/bin/python -m src.integration.live_smoke --fixture three --root . --confirm-remote-inference

train-goemotions:
	.venv/bin/python src/models/train_goemotions.py

train-model2-benchmark:
	.venv/bin/python src/models/train_model2_benchmark.py

train-model2-improved:
	.venv/bin/python src/models/train_model2_improved.py

train-model2-facets:
	.venv/bin/python src/models/train_model2_facets.py

evaluate-model2-selected:
	.venv/bin/python src/models/evaluate_model2_selected.py --confirm-final-test

train-model1-piastra:
	.venv/bin/python src/models/score_model1_piastra.py --env-file .env --confirm-remote-inference

train-model1-piastra-test:
	.venv/bin/python -u src/models/score_model1_piastra.py --env-file .env --confirm-remote-inference --essays data/processed/model1/essays_test.csv --split test --predictions outputs/model-evaluation/model1_piastra_test_predictions.csv --report reports/model-evaluation/model1-piastra-test-benchmark-report.md --metadata data/metadata/model1-piastra-test-benchmark.json --partial-predictions outputs/model-evaluation/model1_piastra_test_predictions.partial.csv

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

train-model1-luna-retrieval:
	.venv/bin/python -u src/models/score_model1_luna_retrieval.py --confirm-remote-inference --max-estimated-usd 0.50

train-model1-luna-retrieval-medium:
	.venv/bin/python -u src/models/score_model1_luna_retrieval.py --confirm-remote-inference --reasoning-effort medium --max-output-tokens 512 --experiment-name luna_retrieval_medium_reasoning --output outputs/model-evaluation/model1_luna_retrieval_medium_validation_output.jsonl --retrievals outputs/model-evaluation/model1_luna_retrieval_medium_validation_references.csv --predictions outputs/model-evaluation/model1_luna_retrieval_medium_validation_predictions.csv --report reports/model-evaluation/model1-luna-retrieval-medium-report.md --metadata data/metadata/model1-luna-retrieval-medium-validation.json --max-estimated-usd 0.75

train-model1-luna-retrieval-medium-test:
	.venv/bin/python -u src/models/score_model1_luna_retrieval.py --confirm-remote-inference --evaluation-file essays_test.csv --split test --reasoning-effort medium --max-output-tokens 512 --experiment-name luna_retrieval_medium_reasoning --output outputs/model-evaluation/model1_luna_retrieval_medium_test_output.jsonl --retrievals outputs/model-evaluation/model1_luna_retrieval_medium_test_references.csv --predictions outputs/model-evaluation/model1_luna_retrieval_medium_test_predictions.csv --report reports/model-evaluation/model1-luna-retrieval-medium-test-report.md --metadata data/metadata/model1-luna-retrieval-medium-test.json --piastra-predictions outputs/model-evaluation/model1_piastra_test_predictions.csv --max-estimated-usd 0.75

train-model1-luna-trait-balanced:
	.venv/bin/python -u src/models/score_model1_luna_retrieval.py --confirm-remote-inference --retrieval-mode trait_balanced --reasoning-effort medium --max-output-tokens 512 --experiment-name luna_trait_balanced_retrieval --output outputs/model-evaluation/model1_luna_trait_balanced_validation_output.jsonl --retrievals outputs/model-evaluation/model1_luna_trait_balanced_validation_references.csv --predictions outputs/model-evaluation/model1_luna_trait_balanced_validation_predictions.csv --report reports/model-evaluation/model1-luna-trait-balanced-validation-report.md --metadata data/metadata/model1-luna-trait-balanced-validation.json --max-estimated-usd 2.00

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
