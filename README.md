[![Build Status](https://travis-ci.com/catalyst-team/bert.svg?branch=master)](https://travis-ci.com/catalyst-team/bert)
[![Telegram](./pics/telegram.svg)](https://t.me/catalyst_team)
[![Gitter](https://badges.gitter.im/catalyst-team/community.svg)](https://gitter.im/catalyst-team/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Slack](./pics/slack.svg)](https://opendatascience.slack.com/messages/CGK4KQBHD)
[![Donate](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/patreon.png)](https://www.patreon.com/catalyst_team)

# Catalyst.Bert

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
