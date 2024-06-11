#!/bin/bash

python -m llava.train.llava_uhd.model_vqa_loader_uhd \
    --model-path ./checkpoints_new/llava-uhd_adapt_1 \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

    # --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl
