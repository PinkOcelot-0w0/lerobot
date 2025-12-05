#训练 SmolVLA 策略
lerobot-train --dataset.repo_id=PinkOcelot/lekiwi_vladata --policy.type=smolvla --policy.empty_cameras=1 --batch_size=16 --steps=30000 --output_dir=outputs/smolvla --job_name=my_smolvla_training --policy.device=cuda --wandb.enable=false --policy.repo_id=pinkocelot/smolvla-latest --save_freq=5000
#训练 ACT 策略
lerobot-train --dataset.repo_id=PinkOcelot/lekiwi_vladata --policy.type=act --output_dir=outputs/act_model --job_name=act --policy.device=cuda --wandb.enable=false --policy.repo_id=${HF_USER}/act_policy --batch_size=8 --steps=100000 --save_freq=10000
#启动策略服务
python -m lerobot.async_inference.policy_server --host=192.168.1.104 --port=8080
#合并数据集
lerobot-edit-dataset --repo_id PinkOcelot/lekiwi_box_merged --operation.type merge --operation.repo_ids "['PinkOcelot/lekiwi_box', 'PinkOcelot/lekiwi_box']"