# run training


# 1.5B
```bash
python train.py  --gradient_checkpointing   --precision fp8   --per_device_train_batch_size 12   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct --dataset_name vinhpx/math_natural_reasoning --with_tracking --report_to wandb --wandb_project Math-reasoning-mixed-fp8
```
bf16
```bash
python train.py  --gradient_checkpointing   --precision bf16  --per_device_train_batch_size 12   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct --dataset_name vinhpx/math_natural_reasoning --with_tracking --report_to wandb --wandb_project Math-reasoning-mixed-bf16
```
fp16
```bash
python train.py  --gradient_checkpointing   --precision fp16  --per_device_train_batch_size 12   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct --dataset_name vinhpx/math_natural_reasoning --with_tracking --report_to wandb --wandb_project Math-reasoning-mixed-fp16
```
fp32
```bash
python train.py  --gradient_checkpointing   --precision fp32  --per_device_train_batch_size 8   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct --dataset_name vinhpx/math_natural_reasoning --with_tracking --report_to wandb --wandb_project Math-reasoning-mixed-fp32
```


# 7b
```bash
accelerate launch --mixed_precision fp8 train.py --use_deepspeed   --zero_stage 3   --gradient_checkpointing   --precision fp8   --per_device_train_batch_size 1   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-7B-Instruct --dataset_name vinhpx/function-calling-mixed --with_tracking --report_to wandb --wandb_project Qwen2.5-7B-Instruct-function-calling-mixed-fp8
python train.py   --zero_stage 3   --gradient_checkpointing   --precision fp8   --per_device_train_batch_size 4   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-7B-Instruct --dataset_name vinhpx/function-calling-mixed --with_tracking --report_to wandb --wandb_project Qwen2.5-7B-Instruct-function-calling-mixed-fp8
```

```bash
python train.py   --zero_stage 3   --gradient_checkpointing   --precision bf16  --per_device_train_batch_size 2   --gradient_accumulation_steps 16 --model_name_or_path Qwen/Qwen2.5-7B-Instruct --dataset_name vinhpx/function-calling-mixed --with_tracking --report_to wandb --wandb_project Qwen2.5-7B-Instruct-function-calling-mixed-bf16
```
