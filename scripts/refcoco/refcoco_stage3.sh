python main.py \
    --model gpt-3.5-turbo-instruct \
    --dataset refcoco \
    --coco_dir /path/to/coco/ \
    --test_num 100 \
	--stage 3 \
    --inference_prompt_path prompts/refcoco/refcoco_stage3.prompt \
    --dataset_dir /path/to/refcoco/ \
    --save_output \
    --output_dir save/results/refcoco/refcoco_stage3/