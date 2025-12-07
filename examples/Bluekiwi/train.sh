lerobot-train --dataset.repo_id=PinkOcelot/lekiwi_vladata --policy.type=smolvla --policy.load_vlm_weights=true --policy.train_expert_only=true --policy.empty_cameras=1 --policy.max_action_dim=32 --batch_size=8 --steps=20000 --output_dir=outputs/smolvla --job_name=my_smolvla_training --policy.device=cuda --wandb.enable=false --policy.repo_id=pinkocelot/smolvla-latest --save_freq=4000

lerobot-train --dataset.repo_id=PinkOcelot/lekiwi_vladata --policy.type=act --output_dir=outputs/act_model --job_name=act --policy.device=cuda --wandb.enable=false --policy.repo_id=${HF_USER}/act_policy --batch_size=8 --steps=100000 --save_freq=10000

python -m lerobot.async_inference.policy_server --host=192.168.31.201 --port=8080