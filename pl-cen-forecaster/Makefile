.PHONY: raw data panel eval eval-ext
raw:
	python -m src.build_dataset --raw-only
data:
	python -m src.build_dataset
panel:
	python -m src.build_dataset --from-raw
eval:
	python -m src.evaluate
eval-ext:
	python -m src.evaluate --extended
