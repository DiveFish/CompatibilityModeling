#!/bin/bash

TRAIN_FILE=$1
DEV_FILE=$2

WORD_EMBEDS=$3
POS_TAG_EMBEDS=$4

# AVERAGE OVER ALL NOUNS (INCLUDING OTHER AND AMBIGUOUS).

# Threshold at 2.
python3 -m src.pp_head_selection.train $TRAIN_FILE $DEV_FILE cand-score -w $WORD_EMBEDS -p $POS_TAG_EMBEDS -v -o -a -t 2 -n 150
# Threshold at default == 3.
python3 -m src.pp_head_selection.train $TRAIN_FILE $DEV_FILE cand-score -w $WORD_EMBEDS -p $POS_TAG_EMBEDS -v -o -a -n 150
# Threshold at 4.
python3 -m src.pp_head_selection.train $TRAIN_FILE $DEV_FILE cand-score -w $WORD_EMBEDS -p $POS_TAG_EMBEDS -v -o -a -t 4 -n 150


# AVERAGE OVER NOUNS AND AMBIGUOUS NOUNS.
# Threshold at 2.
python3 -m src.pp_head_selection.train $TRAIN_FILE $DEV_FILE cand-score -w $WORD_EMBEDS -p $POS_TAG_EMBEDS -v -a -t 2 -n 150
# Threshold at 3.
python3 -m src.pp_head_selection.train $TRAIN_FILE $DEV_FILE cand-score -w $WORD_EMBEDS -p $POS_TAG_EMBEDS -v -a -n 150
# Threshold at 4.
python3 -m src.pp_head_selection.train $TRAIN_FILE $DEV_FILE cand-score -w $WORD_EMBEDS -p $POS_TAG_EMBEDS -v -a -t 4 -n 150

# AVERAGE OVER NOUNS AND OTHER NOUNS.
python3 -m src.pp_head_selection.train $TRAIN_FILE $DEV_FILE cand-score -w $WORD_EMBEDS -p $POS_TAG_EMBEDS -v -o -n 150


# AVERAGE OVER NOUNS THAT ARE NOT IN OTHER CATEGORY OR AMBIGUOUS.
python3 -m src.pp_head_selection.train $TRAIN_FILE $DEV_FILE cand-score -w $WORD_EMBEDS -p $POS_TAG_EMBEDS -v -n 150
