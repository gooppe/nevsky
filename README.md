# Simple and Light Translation Bot

[![Build Status](https://travis-ci.com/gooppe/nevsky.svg?branch=master)](https://travis-ci.com/gooppe/nevsky)

- [Simple and Light Translation Bot](#simple-and-light-translation-bot)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Documentation](#documentation)
    - [Environment variables](#environment-variables)
    - [Installing dictionary](#installing-dictionary)
    - [Installing models](#installing-models)
  - [Usage](#usage)
    - [Command line interface](#command-line-interface)
    - [Running local bot](#running-local-bot)

## Requirements

- python >= 3.6
- pytorch == 1.4
- psycopg2

## Installation
Local installation:
```bash
pip install git+https://github.com/gooppe/nevsky
```

Using docker:
```bash
git clone https://github.com/gooppe/nevsky.git
docker build -t nevsky nevsky/
```

## Documentation
### Environment variables
In order to launch the bot, you must specify two variables: `TELEGRAM_TOKEN` and `DATABASE_URL`. Database url is a *PosgreSQL* connection string. During the deployment to *Heroku*, these configuration variables are set via the control panel or generated automatically when using Postgres add-ons.

### Installing dictionary                                                                         |

In order to support dictionary-based translation package requires PostgreSQL database. It is possible to export lang dictionary using [data.py](nevsky/data.py) script:

```bash
python data.py --df dictionary.xdxf --table ru_en
```

### Installing models
Install pretrained translation model:
```bash
nevsky download ru_en_50
```

## Usage
### Command line interface

Translate sentence via command line:
```bash
nevsky translate dumps/ru_en_50 'Привет, Мир!'
```

### Running local bot
In order to run bot localy:
```bash
nevsky bot ru_en_50
```