import os
from argparse import ArgumentParser

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from models.pl_to_model import TopologyOptimizationModel  # 导入 TO 模型类

# 设置 WANDB_API_KEY 环境变量
os.environ["WANDB_API_KEY"] = "66090b2e30fe0158dbba101f2e57e4e23dd4da4f"
# 设置 WANDB_MODE 环境变量
os.environ["WANDB_MODE"] = "online"  # 或者 "offline", "dryrun", "disabled"

def arg_parser():
    parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TO dataset.')
    parser.add_argument('--storage_path', type=str, default="runs", required=True, help="存储路径")
    parser.add_argument('--data_pkl', type=str, required=True, help="数据集的.pkl文件路径")

    parser.add_argument('--batch_size', type=int, default=64, help="训练时的批次大小")
    parser.add_argument('--num_epochs', type=int, default=50, help="训练的总轮数")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="优化器的学习率")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="权重衰减,L2正则化系数")
    parser.add_argument('--lr_scheduler', type=str, default='constant', help="学习率调度器类型")

    parser.add_argument('--num_workers', type=int, default=16, help="加载数据时的工作线程数")
    parser.add_argument('--fp16', action='store_true', help="是否使用16位浮点数进行训练")
    parser.add_argument('--use_activation_checkpoint', action='store_true', help="是否使用激活检查点来节省内存")

    parser.add_argument('--diffusion_type', type=str, default='gaussian', help="扩散模型的类型gaussian或categorical")
    parser.add_argument('--diffusion_schedule', type=str, default='linear', help="扩散调度类型,如 'linear' ")
    parser.add_argument('--diffusion_steps', type=int, default=1000, help="扩散过程中的步骤数")
    parser.add_argument('--inference_diffusion_steps', type=int, default=1000, help="推理时的扩散步骤数")
    parser.add_argument('--inference_schedule', type=str, default='linear', help="推理时的扩散调度类型")
    parser.add_argument('--inference_trick', type=str, default="ddim", help="推理时使用的技巧,如 'ddim' ")
    parser.add_argument('--sequential_sampling', type=int, default=1, help="顺序采样的次数")
    parser.add_argument('--parallel_sampling', type=int, default=1, help="并行采样的次数")

    parser.add_argument('--n_layers', type=int, default=12, help="图神经网络的层数")
    parser.add_argument('--hidden_dim', type=int, default=256, help="隐藏层的维度大小")
    parser.add_argument('--sparse_factor', type=int, default=-1, help="稀疏因子，控制图的稀疏程度")
    parser.add_argument('--aggregation', type=str, default='sum', help="图神经网络中的聚合方式,如sum")
    parser.add_argument('--two_opt_iterations', type=int, default=1000, help="二次优化2-opt算法的迭代次数")
    parser.add_argument('--save_numpy_heatmap', action='store_true', help="是否保存Numpy热力图")

    parser.add_argument('--project_name', type=str, default='to_diffusion', help="项目名称")
    parser.add_argument('--wandb_entity', type=str, default=None, help="Weights and Biases项目的实体名称")
    parser.add_argument('--wandb_logger_name', type=str, default=None, help="Weights and Biases的日志名称")
    parser.add_argument("--resume_id", type=str, default=None, help="继续训练时在wandb上的ID")
    parser.add_argument('--ckpt_path', type=str, default=None, help="模型检查点的路径")
    parser.add_argument('--resume_weight_only', action='store_true', help="是否仅恢复模型权重而不是继续训练")
    parser.add_argument('--device', type=str, default='0', help="选择使用哪一张或哪几张显卡:0,1,2,3,cpu")


    parser.add_argument('--do_train', action='store_true', help="是否进行训练")
    parser.add_argument('--do_test', action='store_true', help="是否进行测试")
    parser.add_argument('--do_valid_only', action='store_true', help="是否仅进行验证")
    # 添加默认值为100的validation_examples参数
    parser.add_argument('--validation_examples', type=int, default=100, help="验证集的样本数量")

    args = parser.parse_args()
    return args

def main(args):
    epochs = args.num_epochs
    project_name = args.project_name

    model_class = TopologyOptimizationModel
    saving_mode = 'max'

    model = model_class(param_args=args)

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path, 'models'),
        id=args.resume_id or wandb_id,
    )
    rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")

    checkpoint_callback = ModelCheckpoint(
        monitor='val/solved_cost', mode=saving_mode,
        save_top_k=3, save_last=True,
        dirpath=os.path.join(wandb_logger.save_dir,
                             args.wandb_logger_name or 'default',
                             wandb_logger.version,
                             'checkpoints'),
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        max_epochs=epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(static_graph=True),
        precision=16 if args.fp16 else 32,
    )

    rank_zero_info(
        f"{'-' * 100}\n"
        f"{str(model.model)}\n"
        f"{'-' * 100}\n"
    )

    ckpt_path = args.ckpt_path

    if args.do_train:
        if args.resume_weight_only:
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)

        if args.do_test:
            trainer.test(model, ckpt_path=checkpoint_callback.best_model_path)

    elif args.do_test:
        trainer.validate(model, ckpt_path=ckpt_path)
        if not args.do_valid_only:
            trainer.test(model, ckpt_path=ckpt_path)
    trainer.logger.finalize("success")

if __name__ == '__main__':
    args = arg_parser()
    main(args)