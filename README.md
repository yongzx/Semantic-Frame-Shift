# README

### Dependencies

The program is run with Python 3.6 with the following dependencies:

- `torch-geometric`
- `flair`
- `laserembeddings`
- `dataclasses`
- `dill`

### Datasets
The semantic frame shift datasets are found in the `data/` folder. Here are the descriptions of each file:

- `data/en-de.results.txt`: Annotation pairs of English-German parallel sentences with the diverging semantic frame labels
- `data/en-de.same.results.txt`: Annotation pairs of English-German parallel sentences with the same semantic frame labels
- `data/en-pt.same.results.txt`: Annotation pairs of English-Brazilian Portuguese parallel sentences with the same semantic frame labels
- `data/en-pt.results.txt`: Annotation pairs of English-Brazilian Portuguese parallel sentences with the diverging semantic frame labels
