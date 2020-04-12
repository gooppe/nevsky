# Simple and Light Translation Bot

[![Build Status](https://travis-ci.com/gooppe/nevsky.svg?branch=master)](https://travis-ci.com/gooppe/nevsky)

## Requirements

- python >= 3.7

## Installation

```bash
pip install git+https://github.com/gooppe/nevsky
```

## Usage

Download pretrained translation model:

| Model             | Link                                                                           |
| ----------------- | ------------------------------------------------------------------------------ |
| ru -> en, *small* | [ru_en_50](https://drive.google.com/open?id=1dbDou2VN2GEFF7kEfR7rhKO25smq1Cv1) |

Extract dump archive and run `nevsky translate`

```bash
nevsky translate ru_en_50 'Привет, Мир!'
```

The first argument is a dump folder, the second is an input sentence.