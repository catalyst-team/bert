
# bert-ner-catalyst-starter

A barebones (Distil)BERT pipeline for token classification tasks driven by [catalyst](https://github.com/catalyst-team/catalyst).


## Getting started

- In your virtual environment run
```
    pip install -e .
```
- Check [experiment.py](bert_ner/experiment.py) for loading train/test data. At the moment the pipeline assumes two JSON lines files containing `['content', 'tagged_attributes']` columns, where `tagged_attributes` is a list of substrings in `content`.
- Possibly modify [dataset.py](bert_ner/dataset.py) to suit your data preprocessing needs. The pipeline makes assumption that there are two classes of tokens.
- Start training your model
```
catalyst-dl run -C bert_ner/config.yml
```

## Monitoring

Run the following command to see metrics in Tensorboard
```
    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
```
