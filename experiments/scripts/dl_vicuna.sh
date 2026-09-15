#!/bin/bash
HF="${HF:-hf}"
# serialize: let Llama (Ultravox arm) finish first so that arm can run while Vicuna downloads
until grep -q LLAMA_DONE ~/dl_ultravox.log 2>/dev/null; do sleep 30; done
$HF download lmsys/vicuna-13b-v1.1 > ~/dl_vicuna.log 2>&1 && echo VICUNA_DONE >> ~/dl_vicuna.log
