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

### Pretrained Models
Since we evaluate our model with a 5-fold Nested Cross Validation instead of train/validation/test split, our pretrained semantic frame embeddings is the average of all the five trained models. 

The saved pretrained models is in `pretrained/model_and_embeddings.pt` and you can run `python3 load_model.py` on GPU to obtain the semantic frame embeddings.

### Evaluation
The folder `evaluation/` contains two Jupyter notebooks that illustrate the five-fold nested cross validation evaluation of our proposed model and the UMAP representations of semantic frame embeddings.

- `evaluation/Training_and_Evaluation_Multi_task_Learning_GAT_FrameNet.ipynb`: Training and evaluation of GAT models (with auxiliary tasks and five-fold nested cross-validation)
- `evaluation/UMAP_frame_embeddings.ipynb`: UMAP representations of semantic frame embeddings with pre-trained models.
