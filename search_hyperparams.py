import optuna
import os
import subprocess
import json
import glob
import re
from mmcv import Config  # 旧版使用 mmcv.Config，新版是 mmengine.Config

def parse_mmseg_table(log_output):
    """
    专门解析 MMSegmentation 的 ASCII 表格日志
    返回最后一个表格中的 mIoU 值
    """
    lines = log_output.strip().split('\n')
    
    # 倒序遍历，因为我们通常想要最后一次评估的结果（训练结束时的结果）
    # 如果你想要最佳值（Best），通常最后一次评估会包含 best 记录，或者最后一次即为最终结果
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        print(line)
        # 1. 找到表头行 (包含 aAcc, mIoU, mAcc)
        if '|' in line and 'mIoU' in line and 'aAcc' in line:
            # 2. 确定数值所在的行
            # 表头是第 i 行
            # 分隔线是第 i+1 行
            # 数值通常是第 i+2 行
            value_row_index = i + 2
            
            if value_row_index < len(lines):
                val_line = lines[value_row_index]
                
                # val_line 长这样: "| 92.07 | 70.37 | 83.66 |"
                parts = val_line.split('|')
                
                # 分割后 parts 应该是 ['', ' 92.07 ', ' 70.37 ', ' 83.66 ', '']
                # mIoU 在第 3 个位置 (索引为 2)
                if len(parts) >= 3:
                    try:
                        miou_str = parts[2].strip() # 去除空格，拿到 "70.37"
                        return float(miou_str)
                    except ValueError:
                        continue # 如果转换失败，继续找
    
    return 0.0 # 如果没找到，返回 0

def objective(trial):
    # 1. 读取你的基础配置文件
    cfg_path = './configs/voc_weakclip_vit-b_512x512_20k_mct.py' # 替换为你的实际文件名
    cfg = Config.fromfile(cfg_path)
    # 2. === 定义搜索空间 ===
    
    # A. 搜索基础学习率 (注意：通常在 optimizer 中，虽然你看不到它，但它在 _base_ 里)
    # 我们直接在顶层覆盖它
    current_lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    if cfg.optimizer.get('lr') is None:
        # 如果当前 optimizer 配置里没有 lr (可能在 constructor 里)，我们需要确保加上
        cfg.optimizer['lr'] = current_lr
    else:
        cfg.optimizer.lr = current_lr
    
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    cfg.optimizer['weight_decay'] = weight_decay
    # B. 搜索 Backbone 的冻结策略 (重点！)
    # 0.0 = 完全冻结, 0.1 = 慢速微调, 1.0 = 同步训练
    #backbone_lr_mult = trial.suggest_categorical('backbone_lr_mult', [0.0, 0.01, 0.1, 1.0])
    
    # 根据你的 config 结构定位路径
    # 你的 config: optimizer -> paramwise_cfg -> custom_keys -> backbone -> lr_mult
    #cfg.optimizer['paramwise_cfg']['custom_keys']['backbone']['lr_mult'] = backbone_lr_mult
    
    # 同理也可以搜索 text_encoder
    # text_encoder_mult = trial.suggest_categorical('text_mult', [0.0, 0.1])
    # cfg.optimizer['paramwise_cfg']['custom_keys']['text_encoder']['lr_mult'] = text_encoder_mult

    # C. 搜索 Warmup 步数
    #warmup_iters = trial.suggest_int('warmup_iters', 500, 3000, step=500)
    #cfg.lr_config['warmup_iters'] = warmup_iters

    # 3. === 设置运行环境 ===
    
    # 缩短训练时间用于搜索 (例如只跑 2000 iter)
    # 旧版通常叫 runner.max_iters 或 total_iters，具体看你的 _base_/schedule_20k.py
    cfg.runner = dict(type='IterBasedRunner', max_iters=4000) 
    cfg.evaluation = dict(interval=1000, metric='mIoU')# 确保最后做一次评估
    cfg.checkpoint_config = dict(by_epoch=False, interval=2000)

    # 设置独立的 work_dir
    trial_dir = f'work_dirs/optuna_search/trial_{trial.number}'
    cfg.work_dir = trial_dir
    
    # 4. === 保存临时 Config 文件 ===
    temp_config_path = f'temp_config_trial_{trial.number}.py'
    cfg.dump(temp_config_path)

    # 5. === 启动子进程训练 ===
    # 注意：这里使用 tools/train.py，请确保路径正确
    # 如果你是多卡训练，可以使用 bash dist_train.sh ...
    cmd = [
        'python', 'tools/train.py', 
        temp_config_path, 
        '--work-dir', trial_dir,
]
    # 运行并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 2. 找到该目录下最新的 JSON 日志文件
    # MMSeg 会生成类似 20230501_120000.log.json 的文件
    json_logs = glob.glob(os.path.join(trial_dir, '*.log.json'))

    if not json_logs:
        print("❌ 未找到任何日志文件，请检查 work_dir 路径是否正确。")
    else:
        # 按修改时间排序，取最后一个（最新的）
        latest_log = max(json_logs, key=os.path.getmtime)
        print(f"📂 正在读取日志文件: {latest_log}")

        try:
            # 3. 逐行读取 JSON（MMSeg 的 json log 不是标准的整个 json 对象，而是每一行一个 json 对象）
            last_metric = None

            with open(latest_log, 'r') as f:
                for line in f:
                    log_entry = json.loads(line)
                    # 检查这一行是否包含评估指标 (通常包含 'mIoU' 或 'aAcc')
                    if 'mIoU' in log_entry:
                        last_metric = log_entry
            final_miou = last_metric['mIoU']
        except Exception as e:
            print(f"读取日志出错: {e}")

        
    # 删除临时 config
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    return final_miou

if __name__ == '__main__':
    storage_name = "sqlite:///search_result.db"
    study = optuna.create_study(
        study_name="weakclip_opt",
        storage=storage_name, 
        direction='maximize',
        load_if_exists=True
    )
    study.optimize(objective, n_trials=15)
    print("Best:", study.best_params)