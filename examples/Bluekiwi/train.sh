#训练 SmolVLA 策略
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=PinkOcelot/lekiwi_box_merged --dataset.root=/home/pinkocelot/.cache/huggingface/lerobot/pinkocelot/lekiwi_box_merged --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --policy.output_features='{"action": {"shape": [9], "type": "ACTION"}}' --batch_size=12 --steps=30000 --output_dir=outputs/smolvla --job_name=my_smolvla_training --policy.device=cuda --wandb.enable=false --policy.repo_id=pinkocelot/smolvla-latest --save_freq=5000
#训练 ACT 策略
lerobot-train --dataset.repo_id=PinkOcelot/lekiwi_vladata --policy.type=act --output_dir=outputs/act_model --job_name=act --policy.device=cuda --wandb.enable=false --policy.repo_id=${HF_USER}/act_policy --batch_size=8 --steps=100000 --save_freq=10000
#启动策略服务
python -m lerobot.async_inference.policy_server --host=192.168.1.104 --port=8080
python -m lerobot.async_inference.robot_client --server_address=10.20.30.100:8080 --robot.type=lekiwi --robot.port=/dev/ttyACM0 --robot.id=my_awesome_kiwi --robot.cameras="{ front: {type: opencv, index_or_path: \"/dev/video0\", width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: \"/dev/video2\", width: 640, height: 480, fps: 30}}" --task="Grab a tissue and place it in front of the box. Then open the lid, place it on the right side of the box, put the tissue inside, and close the lid again." --policy_type=smolvla --pretrained_name_or_path=models/smolvla --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5
#合并数据集
lerobot-edit-dataset --repo_id PinkOcelot/lekiwi_box_merged --operation.type merge --operation.repo_ids "['PinkOcelot/lekiwi_box', 'PinkOcelot/lekiwi_box']"
