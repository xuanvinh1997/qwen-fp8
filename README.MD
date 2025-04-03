# run training

```bash
accelerate launch --mixed_precision fp8 train.py --use_deepspeed   --zero_stage 3   --gradient_checkpointing   --precision fp8   --per_device_train_batch_size 1   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-7B-Instruct --dataset_name vinhpx/function-calling-mixed --with_tracking --report_to wandb --wandb_project Qwen2.5-7B-Instruct-function-calling-mixed-fp8
python train.py   --zero_stage 3   --gradient_checkpointing   --precision fp8   --per_device_train_batch_size 4   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-7B-Instruct --dataset_name vinhpx/function-calling-mixed --with_tracking --report_to wandb --wandb_project Qwen2.5-7B-Instruct-function-calling-mixed-fp8
```

```bash
python train.py   --zero_stage 3   --gradient_checkpointing   --precision bf16  --per_device_train_batch_size 2   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-7B-Instruct --dataset_name vinhpx/function-calling-mixed --with_tracking --report_to wandb --wandb_project Qwen2.5-7B-Instruct-function-calling-mixed-bf16
```
