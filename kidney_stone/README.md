This sub project contain:

data_exploration.ipnyb - jupyter notebook with some plots - histograms, corelations, scatterplots. Used to choose preprocesing techniques, and traning alghoritms.
For intresting techniques PCA was used, to check linear separability of data.

main.py - entrypoint for application, connect to azure, and launch 2 pipelines.

pipelines.py - definitions of pipelines.

conda.yaml - environment definition, with scikit-learn, I thought about used curated image, but azure images are really not recent, and strange in my opinion.

Other files - particular steps in pipelines.