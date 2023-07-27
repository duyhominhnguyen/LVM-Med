default_target: main
.PHONY : default_target

$(VERBOSE).SILENT:

SHELL = /bin/sh

sam_2D_example:
	python3 medsam.py -c dataloader/yaml_data/buid_sam.yml -2D 
.PHONY: sam_2D_example