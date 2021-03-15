Math Project
========

Author/Contact: Harold Mouchère (harold.mouchere@univ-nantes.fr)

This is a Master Project to apply machine (deep) learning tool on a challenging computer vision problem: recognition of handwritten Math Expression.

The subject is available in the directory `subject/` and the provided code in `code/`. Some toy examples are available in the `data/` directory but the complete dataset is available in the TC11 repository.


## Usage of some provided tools

### convertInkmlToImg.py
	Usage: convertInkmlToImg.py  (path to file or folder) dim padding
		+ (file|folder)        - required str
		+ dim                  - optional int (def = 28)
		+ padding              - optional int (def =  0)

### usefulTools.Py

Use func `load_dataset` to load CRHOME data and create database from image (png) folders if the .npz data file don't exist.
