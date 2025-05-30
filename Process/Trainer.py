from datetime import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import sys
import random
import wandb

sys.path.append(r"/data01/zt/Process")
from PostProcess import calculate_metric_per_video, calculate_metric_simple_inference
from metrics import calculate_metrics_simple

sys.path.append(r"/data01/zt/Loss")
import TorchLossComputer


class BaseProcessor:
    def __init__(self, args, model):
        self.args = args
        self.device = args.device
        self.model = model
        self.target_label = "bvp"
        self.modal_used = self.args.modal_used

    def process_batch(self, batch):
        """Process a single batch of data and return predictions and labels."""
        data_input = {}
        if "front" in self.modal_used and "video_front" in batch["modals"]:
            data_input["front"] = batch["modals"]["video_front"].to(self.device)
        if "back" in self.modal_used and "video_back" in batch["modals"]:
            data_input["back"] = batch["modals"]["video_back"].to(self.device)

        if not data_input:
            raise ValueError("没有可用的视频输入，请检查配置")

        label_ppg = batch["labels"][self.target_label].to(self.device)

        # for multi-modal
        if len(data_input) > 1:
            if self.args.video_backbone == "PhysMamba":
                data_input["front"] = data_input["front"].permute(0,4,1,3,2)
                data_input["back"] = data_input["back"].permute(0,4,1,3,2)

            elif self.args.video_backbone == "RhythmFormer":
                data_input["front"] = data_input["front"].permute(0,1,4,2,3)
                data_input["back"] = data_input["back"].permute(0,1,4,2,3)

            elif self.args.video_backbone == "PhysNet_padding_Encoder_Decoder_MAX":
                data_input["front"] = data_input["front"].permute(0,4,1,3,2)
                data_input["back"] = data_input["back"].permute(0,4,1,3,2)      

            elif self.args.video_backbone == "ViT_ST_ST_Compact3_TDC_gra_sharp":
                data_input["front"] = data_input["front"].permute(0,4,1,2,3)
                data_input["back"] = data_input["back"].permute(0,4,1,2,3)   

            output = self.model(data_input)
            pred_ppg = output["rPPG"]
            
        else:
            modal = self.args.modal_used[0]
            if self.args.video_backbone == "RhythmFormer":
                video_input = data_input[modal].permute(0,1,4,2,3)
                pred_ppg = self.model(video_input)
            elif self.args.video_backbone == "PhysNet_padding_Encoder_Decoder_MAX":
                video_input = data_input[modal].permute(0,4,1,3,2)
                pred_ppg, _ = self.model(video_input)
            elif self.args.video_backbone == "ViT_ST_ST_Compact3_TDC_gra_sharp":
                gra_sharp = 2.0
                video_input = data_input[modal].permute(0,4,1,2,3)
                pred_ppg, _, _, _ = self.model(video_input, gra_sharp)
            elif self.args.video_backbone == "PhysMamba":
                video_input = data_input[modal].permute(0,4,1,3,2)
                pred_ppg = self.model(video_input)
            elif self.args.video_backbone == "PhysMamba_fft":
                video_input = data_input[modal].permute(0,1,4,2,3)
                pred_ppg = self.model(video_input)

        pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / (torch.std(pred_ppg) + 1e-8)
        label_ppg = (label_ppg - torch.mean(label_ppg)) / (torch.std(label_ppg) + 1e-8)

        return pred_ppg, label_ppg

class LossCalculator(BaseProcessor):
    def __init__(self, args, model, latent_loss_weight=0.25):
        super().__init__(args, model)
        self.latent_loss_weight = latent_loss_weight
        self.criterion = TorchLossComputer.RhythmFormer_Loss()
        self.person_criterion = TorchLossComputer.Neg_Pearson() # range[0, 1]
        self.MSE = nn.MSELoss()

    def compute(self, batch):
        """
        Compute the loss of the model on the given batch.
        """
        pred_ppg, label_ppg = self.process_batch(batch)

        loss = 0.0
        N = label_ppg.shape[0]
        for ib in range(N):
            loss = loss + self.criterion(pred_ppg[ib], label_ppg[ib], 0, 30, diff_flag=0)
        loss = loss / N

        return loss

class Trainer():
    def __init__(self, args, model, train_loader, eval_loader, test_loader, verbose=False):
        
        # 基础配置
        self.args = args
        self.verbose = verbose
        self.device = args.device
        print(f"使用设备: {self.device}")
        self.modal_used = self.args.modal_used
        print(f"使用模态:{self.modal_used}")
        
        # 数据加载器
        self.batch_size = args.batch_size
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        # 加载model
        self.model = model

        # 处理器和损失计算器
        self.cr = LossCalculator(self.args,self.model)
        self.processor = BaseProcessor(self.args, self.model)

        # 训练参数
        self.lr = args.lr
        self.num_epoch = args.epochs
        self.num_train_batches = len(train_loader)
        self.step = 0

        self.target_label = "bvp"

        # 优化器和学习率调度器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.lr, epochs=self.num_epoch, steps_per_epoch=self.num_train_batches)
        
        # 测试配置
        if self.args.dataset_name == "tjk_multimodal":
            self.hr_calculate_batch_step = 4
            self.video_Hz = 30
        elif self.args.dataset_name == "anzhen_multimodal":
            self.hr_calculate_batch_step = 4
            self.video_Hz = 30
        elif self.args.dataset_name == "SUMS_multimodal":
            self.hr_calculate_batch_step = 4 
            self.video_Hz = 30
        else:
            raise ValueError("没有可用的测试配置，请检查!")

        self.hr_calculate_step = self.hr_calculate_batch_step * self.args.seq_len
        

    def train(self):
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            wandb.log({"train/loss_epoch": loss_epoch, "epoch": epoch})
            self.print_process(f"Epoch {epoch+1}/{self.num_epoch} | Train/Loss: {loss_epoch:.4f} | Time: {time_cost:.2f}s")
            self.eval_model()
            if (epoch+1) % 5 == 0:
                self.test_model(epoch)

        model_path = "/data01/zt/runs/"+self.args.video_backbone+"_"+self.args.dataset_name+"_model_weights.pth"
        
        self.save_model(model_path)
        # print(f"训练完成! 最佳指标: {self.best_metric:.4f}")
        # return self.best_metric
    
    def _train_one_epoch(self):
        t0 = time.perf_counter()
        
        self.model.train()
        # random.shuffle(self.train_loader)
        pbar = tqdm(self.train_loader, 
                   desc="训练进度",
                   disable= self.verbose,
                   ncols=100,
                   file=sys.stdout)

        loss_sum = 0
        for idx, batch in enumerate(pbar):
            loss = self.cr.compute(batch)
            loss_sum += loss.item()
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            wandb.log({"train/loss": loss.item()}, step=self.step)

            self.step += 1
        return loss_sum / (idx + 1), time.perf_counter() - t0

    def eval_model(self):
        self.model.eval()
        pbar = tqdm(self.eval_loader,
                   desc="评估进度",
                   disable=self.verbose,
                   ncols=100,
                   file=sys.stdout)

        valid_loss_sum = []
        with torch.no_grad():
            for idx, batch in enumerate(pbar):
                valid_loss = self.cr.compute(batch)
                valid_loss_sum.append(valid_loss.item())

        eval_loss = np.mean(np.asarray(valid_loss_sum))
        wandb.log({"valid/Loss": eval_loss}, step=self.step)
        print(f"valid/Loss: {eval_loss}")

    def test_model(self, epoch):
        print("Testing!")
        pbar = tqdm(self.test_loader,
            desc="测试进度",
            disable=self.verbose,
            ncols=100,
            file=sys.stdout)
        
        calculate_time = (self.args.seq_len / self.video_Hz) * self.hr_calculate_batch_step
        
        # ["MAE","RMSE","MAPE","Pearson"]
        metrics_total = {key: [] for key in {"FFT_MAE", "FFT_RMSE", "FFT_MAPE", "FFT_Pearson"}}

        all_pred_ppg = []
        all_label_ppg = []

        pred_hr_fft_list = []
        gt_hr_fft_list = []

        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(pbar):

                pred_ppg, label_ppg = self.processor.process_batch(batch) # [N,L]
                # Store the data before processing
                all_pred_ppg.extend([pred_ppg[i].cpu() for i in range(pred_ppg.shape[0])])
                all_label_ppg.extend([label_ppg[i].cpu() for i in range(label_ppg.shape[0])])
                
        # When we have enough samples, concatenate and process
        idx = 0
        all_pred_ppg = np.concatenate(all_pred_ppg)
        all_label_ppg = np.concatenate(all_label_ppg)
        total_samples = len(all_pred_ppg)
        while (idx + 1) * self.hr_calculate_step <= total_samples:
            start = idx * self.hr_calculate_step
            end = (idx + 1) * self.hr_calculate_step

            pred_ppg_concat = all_pred_ppg[start:end]
            label_ppg_concat = all_label_ppg[start:end]
            
            # Calculate metrics
            pred_hr_fft, gt_hr_fft = calculate_metric_simple_inference(
                np.array(pred_ppg_concat),
                np.array(label_ppg_concat),
                original_fps = self.video_Hz
            )

            gt_hr_fft_list.append(gt_hr_fft)
            pred_hr_fft_list.append(pred_hr_fft)
            idx +=1

        remaining_samples = total_samples - idx * self.hr_calculate_step
            
        if remaining_samples > 0:
            try:
                start = idx * self.hr_calculate_step
                end = total_samples  # 即 start + remaining_samples

                pred_remaining = all_pred_ppg[start:end]
                label_remaining = all_label_ppg[start:end]

                # 计算剩余样本的指标
                pred_hr_fft, gt_hr_fft = calculate_metric_simple_inference(
                    np.array(pred_remaining),
                    np.array(label_remaining)
                )
                gt_hr_fft_list.append(gt_hr_fft)
                pred_hr_fft_list.append(pred_hr_fft)

            except RuntimeError as e:
                print(f"Warning: 处理剩余 {remaining_samples} 个样本失败（张量堆叠错误）: {e}")
            except Exception as e:
                print(f"Warning: 处理剩余 {remaining_samples} 个样本时发生未知错误: {e}")

        # Create 6x1 subplots
        plt.figure(figsize=(10, 12))
        for i in range(6):
            start_idx = i * 1000
            end_idx = start_idx + 1000
            
            plt.subplot(6, 1, i+1)
            plt.plot(all_pred_ppg[start_idx:end_idx], label='Predicted PPG', alpha=0.7)
            plt.plot(all_label_ppg[start_idx:end_idx], label='Ground Truth PPG', alpha=0.7)
            
            # Add labels and title for each subplot
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
            plt.title(f'Frames {start_idx}-{end_idx}')
            plt.legend()
            
        # Add main title for the entire figure
        plt.suptitle(f'Comparison of Predicted vs Ground Truth PPG (6 segments): Train epoch:{epoch}')
        plt.tight_layout()
        
        # Log the plot to wandb
        wandb.log({"PPG_Comparison": wandb.Image(plt)})

        # 保存曲线图，使用时间戳确保文件名唯一
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = "/data01/zt/runs/3-fold-test-front-totaldataset"
        os.makedirs(dir_path, exist_ok=True)
        # plt.savefig(f'{dir_path}/BVP_comparison_{self.args.dataset_name}_{timestamp}.png', dpi=300, bbox_inches='tight')
        # Close the plot to free memory
        plt.close()

        # Calculate the metrics
        pred_hr_fft_np = np.array(pred_hr_fft_list)
        gt_hr_fft_np = np.array(gt_hr_fft_list)
        metrics = calculate_metrics_simple(pred_hr_fft_np, gt_hr_fft_np)
            
        for key in metrics_total.keys():
            if metrics.get(key) is not None:
                metrics_total[key].append(metrics.get(key))

        print("metrics:", metrics_total)
        
        wandb.log({
            "test/MAE": metrics_total.get("FFT_MAE")[0],
            "test/RMSE": metrics_total.get("FFT_RMSE")[0],
            "test/MAPE": metrics_total.get("FFT_MAPE")[0],
            "test/Pearson": metrics_total.get("FFT_Pearson")[0],
        })

        # 绘制对比曲线图
        # 创建x轴刻度，每12个样本代表64s
        x_ticks = np.arange(0, len(pred_hr_fft_list)) * calculate_time
        plt.figure(figsize=(10, 6))
        plt.plot(x_ticks, pred_hr_fft_list, label='Predicted Heart Rate')
        plt.plot(x_ticks, gt_hr_fft_list, label='Ground Truth Heart Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Heart Rate (BPM)')
        title_components = [
        f'Heart Rate Comparison, Dataset: {self.args.dataset_name}, Train epoch:{epoch}',
        # f'Heart Rate Comparison Train IDs: {", ".join(map(str, self.args.train_ids))} Test IDs: {", ".join(map(str, self.args.test_ids))}',
        f'Models: {self.args.video_backbone if hasattr(self.args, "video_backbone") else "N/A"} Calculate_time: {calculate_time}s',
        f'Modals: {", ".join(self.args.modal_used) if hasattr(self.args, "modal_used") else "N/A"} HR sample number:{len(pred_hr_fft_list)}',
        f'MAE: {np.nanmean(metrics_total.get("FFT_MAE", [np.nan])):.2f}, RMSE: {np.nanmean(metrics_total.get("FFT_RMSE", [np.nan])):.2f}, MAPE: {np.nanmean(metrics_total.get("FFT_MAPE", [np.nan])):.2f}%, ρ: {np.nanmean(metrics_total.get("FFT_Pearson", [np.nan])):.2f}%',
        ]
        plt.title('\n'.join(filter(None, title_components)), fontsize=12, pad=20)
        plt.legend()
        plt.grid(True)
        # plt.ylim(50, 120)  # 设置合理的y轴范围

        # 保存曲线图，使用时间戳确保文件名唯一
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_ids_str = '_'.join(map(str, self.args.train_ids))
        test_ids_str = '_'.join(map(str, self.args.test_ids))
        dir_path = "/data01/zt/runs/3-fold-test-front-totaldataset"
        os.makedirs(dir_path, exist_ok=True)
        # plt.savefig(f'{dir_path}/heart_rate_comparison_{self.args.dataset_name}_{timestamp}.png', dpi=300, bbox_inches='tight')
        # Log to wandb
        wandb.log({"heart_rate_comparison": wandb.Image(plt)})
        plt.close()

        # Log metrics to TensorBoard
        # for metric_name, metric_value in overall_mean_metrics.items():
        #     self.writer.add_scalar(f'Validation/{metric_name}', metric_value, self.step)

    def save_model(self, path="model_weights.pth"):
        """Save the model weights to a file."""
        torch.save(self.model.state_dict(), path)
        self.print_process(f"Model saved to {path}")    

    def print_process(self, *x):
        if True:
            print(*x)

class TrainerInterable(Trainer):
    def __init__(self, args, model, train_loader, eval_loader, test_loader, verbose=False):
        super().__init__(args, model, train_loader, eval_loader, test_loader, verbose)

    def train(self):
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            wandb.log({"train/loss_epoch": loss_epoch, "epoch": epoch})
            self.print_process(f"Epoch {epoch+1}/{self.num_epoch} | Train/Loss: {loss_epoch:.4f} | Time: {time_cost:.2f}s")
            self.eval_model()
            if (epoch+1) % 5 == 0:
                self.test_model(epoch)

        model_path = "/data01/zt/runs/"+self.args.video_backbone+"_"+self.args.dataset_name+"_model_weights.pth"
        
        self.save_model(model_path)
        # print(f"训练完成! 最佳指标: {self.best_metric:.4f}")
        # return self.best_metric
    
    def _train_one_epoch(self):
        t0 = time.perf_counter()
        
        self.model.train()
        # random.shuffle(self.train_loader)
        pbar = tqdm(len(self.train_loader), 
                   desc="训练进度",
                   disable= self.verbose,
                   ncols=100,
                   file=sys.stdout)

        loss_sum = 0
        for idx, batch in enumerate(self.train_loader):
            loss = self.cr.compute(batch)
            loss_sum += loss.item()
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            wandb.log({"train/loss": loss.item()}, step=self.step)
            self.step += 1
            pbar.update(1)

        return loss_sum / (idx + 1), time.perf_counter() - t0

    def eval_model(self):
        self.model.eval()
        pbar = tqdm(len(self.eval_loader),
                   desc="评估进度",
                   disable=self.verbose,
                   ncols=100,
                   file=sys.stdout)

        valid_loss_sum = []
        with torch.no_grad():
            for idx, batch in enumerate(self.eval_loader):
                valid_loss = self.cr.compute(batch)
                valid_loss_sum.append(valid_loss.item())
                pbar.update(1)

        eval_loss = np.mean(np.asarray(valid_loss_sum))
        wandb.log({"valid/Loss": eval_loss}, step=self.step)
        print(f"valid/Loss: {eval_loss}")

    def test_model(self, epoch):
        print("Testing!")
        pbar = tqdm(len(self.test_loader),
            desc="测试进度",
            disable=self.verbose,
            ncols=100,
            file=sys.stdout)
        
        calculate_time = (self.args.seq_len / self.args.target_hz) * self.hr_calculate_batch_step
        
        # ["MAE","RMSE","MAPE","Pearson"]
        metrics_total = {key: [] for key in {"FFT_MAE", "FFT_RMSE", "FFT_MAPE", "FFT_Pearson"}}

        all_pred_ppg = []
        all_label_ppg = []

        pred_hr_fft_list = []
        gt_hr_fft_list = []

        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(self.test_loader):

                pred_ppg, label_ppg = self.processor.process_batch(batch) # [N,L]
                # Store the data before processing
                all_pred_ppg.extend([pred_ppg[i].cpu() for i in range(pred_ppg.shape[0])])
                all_label_ppg.extend([label_ppg[i].cpu() for i in range(label_ppg.shape[0])])
                pbar.update(1)
                
        # When we have enough samples, concatenate and process
        idx = 0
        all_pred_ppg = np.concatenate(all_pred_ppg)
        all_label_ppg = np.concatenate(all_label_ppg)
        total_samples = len(all_pred_ppg)
        while (idx + 1) * self.hr_calculate_step <= total_samples:
            start = idx * self.hr_calculate_step
            end = (idx + 1) * self.hr_calculate_step

            pred_ppg_concat = all_pred_ppg[start:end]
            label_ppg_concat = all_label_ppg[start:end]
            
            # Calculate metrics
            pred_hr_fft, gt_hr_fft = calculate_metric_simple_inference(
                np.array(pred_ppg_concat),
                np.array(label_ppg_concat),
                original_fps = self.video_Hz
            )

            gt_hr_fft_list.append(gt_hr_fft)
            pred_hr_fft_list.append(pred_hr_fft)
            idx +=1

        remaining_samples = total_samples - idx * self.hr_calculate_step
            
        if remaining_samples > 0:
            try:
                start = idx * self.hr_calculate_step
                end = total_samples  # 即 start + remaining_samples

                pred_remaining = all_pred_ppg[start:end]
                label_remaining = all_label_ppg[start:end]

                # 计算剩余样本的指标
                pred_hr_fft, gt_hr_fft = calculate_metric_simple_inference(
                    np.array(pred_remaining),
                    np.array(label_remaining)
                )
                gt_hr_fft_list.append(gt_hr_fft)
                pred_hr_fft_list.append(pred_hr_fft)

            except RuntimeError as e:
                print(f"Warning: 处理剩余 {remaining_samples} 个样本失败（张量堆叠错误）: {e}")
            except Exception as e:
                print(f"Warning: 处理剩余 {remaining_samples} 个样本时发生未知错误: {e}")

        # Create 6x1 subplots
        plt.figure(figsize=(10, 12))
        for i in range(6):
            start_idx = i * 1000
            end_idx = start_idx + 1000
            
            plt.subplot(6, 1, i+1)
            plt.plot(all_pred_ppg[start_idx:end_idx], label='Predicted PPG', alpha=0.7)
            plt.plot(all_label_ppg[start_idx:end_idx], label='Ground Truth PPG', alpha=0.7)
            
            # Add labels and title for each subplot
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
            plt.title(f'Frames {start_idx}-{end_idx}')
            plt.legend()
            
        # Add main title for the entire figure
        plt.suptitle(f'Comparison of Predicted vs Ground Truth PPG (6 segments): Train epoch:{epoch}')
        plt.tight_layout()
        
        # Log the plot to wandb
        wandb.log({"PPG_Comparison": wandb.Image(plt)})

        # 保存曲线图，使用时间戳确保文件名唯一
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = "/data01/zt/runs/3-fold-test-front-totaldataset"
        os.makedirs(dir_path, exist_ok=True)
        # plt.savefig(f'{dir_path}/BVP_comparison_{self.args.dataset_name}_{timestamp}.png', dpi=300, bbox_inches='tight')
        # Close the plot to free memory
        plt.close()

        # Calculate the metrics
        pred_hr_fft_np = np.array(pred_hr_fft_list)
        gt_hr_fft_np = np.array(gt_hr_fft_list)
        metrics = calculate_metrics_simple(pred_hr_fft_np, gt_hr_fft_np)
            
        for key in metrics_total.keys():
            if metrics.get(key) is not None:
                metrics_total[key].append(metrics.get(key))

        print("metrics:", metrics_total)
        
        wandb.log({
            "test/MAE": metrics_total.get("FFT_MAE")[0],
            "test/RMSE": metrics_total.get("FFT_RMSE")[0],
            "test/MAPE": metrics_total.get("FFT_MAPE")[0],
            "test/Pearson": metrics_total.get("FFT_Pearson")[0],
        })

        # 绘制对比曲线图
        # 创建x轴刻度，每12个样本代表64s
        x_ticks = np.arange(0, len(pred_hr_fft_list)) * calculate_time
        plt.figure(figsize=(10, 6))
        plt.plot(x_ticks, pred_hr_fft_list, label='Predicted Heart Rate')
        plt.plot(x_ticks, gt_hr_fft_list, label='Ground Truth Heart Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Heart Rate (BPM)')
        title_components = [
        f'Heart Rate Comparison, Dataset: {self.args.dataset_name}, Train epoch:{epoch}',
        # f'Heart Rate Comparison Train IDs: {", ".join(map(str, self.args.train_ids))} Test IDs: {", ".join(map(str, self.args.test_ids))}',
        f'Models: {self.args.video_backbone if hasattr(self.args, "video_backbone") else "N/A"} Calculate_time: {calculate_time}s',
        f'Modals: {", ".join(self.args.modal_used) if hasattr(self.args, "modal_used") else "N/A"} HR sample number:{len(pred_hr_fft_list)}',
        f'MAE: {np.nanmean(metrics_total.get("FFT_MAE", [np.nan])):.2f}, RMSE: {np.nanmean(metrics_total.get("FFT_RMSE", [np.nan])):.2f}, MAPE: {np.nanmean(metrics_total.get("FFT_MAPE", [np.nan])):.2f}%, ρ: {np.nanmean(metrics_total.get("FFT_Pearson", [np.nan])):.2f}%',
        ]
        plt.title('\n'.join(filter(None, title_components)), fontsize=12, pad=20)
        plt.legend()
        plt.grid(True)
        # plt.ylim(50, 120)  # 设置合理的y轴范围

        # 保存曲线图，使用时间戳确保文件名唯一
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_ids_str = '_'.join(map(str, self.args.train_ids))
        test_ids_str = '_'.join(map(str, self.args.test_ids))
        dir_path = "/data01/zt/runs/3-fold-test-front-totaldataset"
        os.makedirs(dir_path, exist_ok=True)
        # plt.savefig(f'{dir_path}/heart_rate_comparison_{self.args.dataset_name}_{timestamp}.png', dpi=300, bbox_inches='tight')
        # Log to wandb
        wandb.log({"heart_rate_comparison": wandb.Image(plt)})
        plt.close()

        # Log metrics to TensorBoard
        # for metric_name, metric_value in overall_mean_metrics.items():
        #     self.writer.add_scalar(f'Validation/{metric_name}', metric_value, self.step)



