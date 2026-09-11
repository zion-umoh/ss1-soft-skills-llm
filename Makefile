.PHONY: audit-data watch-data prepare-recruitview extract-recruitview-audio extract-recruitview-llm-features extract-recruitview-speech-embeddings train-model1-feature-fusion analyse-recruitview-feature-balance integrate-recruitview-bessi compare-recruitview-paper prepare-bfi2-bessi prepare-bfi2-bessi-facets train-model2-benchmark train-model2-improved train-model2-facets evaluate-model2-selected test

audit-data:
	.venv/bin/python src/data/audit_raw_datasets.py

watch-data:
	.venv/bin/python src/data/watch_raw_datasets.py

prepare-recruitview:
	.venv/bin/python src/data/prepare_recruitview.py

extract-recruitview-audio:
	.venv/bin/python src/data/extract_recruitview_audio.py

extract-recruitview-llm-features:
	.venv/bin/python src/data/extract_recruitview_llm_features.py

extract-recruitview-speech-embeddings:
	.venv/bin/python -m src.data.extract_recruitview_speech_embeddings

train-model1-feature-fusion:
	.venv/bin/python -m src.models.train_model1_feature_fusion

.PHONY: train-model1-gated-fusion
train-model1-gated-fusion:
	.venv/bin/python -m src.models.train_model1_gated_fusion

.PHONY: close-recruitview-study
close-recruitview-study:
	.venv/bin/python -m src.evaluation.close_recruitview_study

analyse-recruitview-feature-balance:
	.venv/bin/python -m src.evaluation.analyse_feature_balance

integrate-recruitview-bessi:
	.venv/bin/python -m src.evaluation.integrate_recruitview_bessi

compare-recruitview-paper:
	.venv/bin/python -m src.evaluation.compare_recruitview_paper

prepare-bfi2-bessi:
	.venv/bin/python src/data/prepare_bfi2_bessi.py

prepare-bfi2-bessi-facets:
	.venv/bin/python src/data/prepare_bfi2_bessi_facets.py

train-model2-benchmark:
	.venv/bin/python src/models/train_model2_benchmark.py

train-model2-improved:
	.venv/bin/python src/models/train_model2_improved.py

train-model2-facets:
	.venv/bin/python src/models/train_model2_facets.py

evaluate-model2-selected:
	.venv/bin/python src/models/evaluate_model2_selected.py --confirm-final-test

test:
	.venv/bin/python -m unittest discover -s tests
