#!/bin/bash
# background model downloads for the frozen arms
HF="${HF:-hf}"
cd ~
( $HF download fixie-ai/ultravox-v0_6-llama-3_1-8b && echo UV_DONE \
  && $HF download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*" && echo LLAMA_DONE ) > ~/dl_ultravox.log 2>&1 &
( $HF download tsinghua-ee/SALMONN salmonn_v1.pth && echo SALMONN_CKPT_DONE \
  && $HF download WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2 && echo BEATS_DONE \
  && $HF download openai/whisper-large-v2 --exclude "*.bin" --exclude "*.msgpack" --exclude "*.h5" && echo WHISPER_V2_DONE ) > ~/dl_salmonn.log 2>&1 &
wait
echo ALL_DL_DONE
