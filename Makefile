.PHONY: load-seed save-seed

load-seed:
	python scripts/manage_seed.py load

save-seed:
	python scripts/manage_seed.py save
