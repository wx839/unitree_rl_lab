conda activate wx_isaaclab
cd /data3/xwu/ws_loco_policy/unitree_rl_lab
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity


CUDA_VISIBLE_DEVICES=2 nohup python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity > logs/rsl_rl/log/0408/train_29dof_velocity_0408_9.log 2>&1 &

tail -f logs/rsl_rl/train_29dof_velocity.log

tensorboard --logdir logs/rsl_rl/unitree_g1_29dof_velocity


# 断点续训
CUDA_VISIBLE_DEVICES=3 nohup python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --resume --load_run 2026-04-02_09-38-15 --checkpoint model_10500.pt > logs/rsl_rl/train_29dof_resume.log 2>&1 &


# Rough（粗糙地面 + 斜坡）
CUDA_VISIBLE_DEVICES=3 nohup python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Rough \
  > logs/rsl_rl/train_29dof_rough.log 2>&1 &

# Stairs（上下楼梯）
CUDA_VISIBLE_DEVICES=4 nohup python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Stairs \
  > logs/rsl_rl/train_29dof_stairs.log 2>&1 &

# Mixed（全地形混合）
CUDA_VISIBLE_DEVICES=5 nohup python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Mixed \
  > logs/rsl_rl/train_29dof_mixed.log 2>&1 &












unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-04-03_03-10-04           flat 重新训
unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-04-03_03-17-41           2026-04-02_09-38-15/model_10500.pt  断点续训