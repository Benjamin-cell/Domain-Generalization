import torch
import higher
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed import dpo_loss
from datetime import datetime
import torch.distributed as dist


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_preference_pairs_cross_domain(model, data_list, seq_len, short_term_len, feature_count,
                                           predict_length, device, rank, max_pairs=200):
    pairs = []
    with torch.no_grad():
        for data in data_list:
            step_size = max(1, (len(data) - seq_len - predict_length) // max_pairs)
            for i in range(0, len(data) - seq_len - predict_length + 1, step_size):
                if len(pairs) >= max_pairs:
                    break

                x_long = data[i:i + seq_len, :feature_count]
                x_short = data[i + seq_len - short_term_len:i + seq_len, :feature_count]
                y_true = data[i + seq_len:i + seq_len + predict_length, -1]

                x_long = torch.tensor(x_long, dtype=torch.float32).unsqueeze(0)
                x_short = torch.tensor(x_short, dtype=torch.float32).unsqueeze(0)
                x_short_padded = torch.nn.functional.pad(x_short, (0, 0, seq_len - short_term_len, 0))

                x_long = x_long.to(device, non_blocking=True)
                x_short_padded = x_short_padded.to(device, non_blocking=True)
                y_true = torch.tensor(y_true, dtype=torch.float32).to(device)

                y_pred_long = model(x_long).squeeze()
                y_pred_short = model(x_short_padded).squeeze()

                error_long = torch.mean(torch.abs(y_pred_long - y_true)).item()
                error_short = torch.mean(torch.abs(y_pred_short - y_true)).item()

                if error_short < error_long:
                    pairs.append((x_short_padded.squeeze(0).cpu(), x_long.squeeze(0).cpu()))
                else:
                    pairs.append((x_long.squeeze(0).cpu(), x_short_padded.squeeze(0).cpu()))

    return pairs


def meta_finetune_with_dpo(rank, model, meta_train_data_list, seq_len, short_term_len, feature_count,
                           predict_length, device, n_inner_steps=3, n_outer_steps=6, lr_inner=0.01, lr_outer=1e-5):
    base_model = model.module if isinstance(model, DDP) else model
    meta_optimizer = torch.optim.Adam(base_model.parameters(), lr=lr_outer)

    batch_size = 32
    accumulation_steps = 4

    for epoch in range(n_outer_steps):
        meta_optimizer.zero_grad()
        meta_loss = 0.0

        for domain_data in meta_train_data_list:
            preference_pairs = generate_preference_pairs_cross_domain(
                model,
                [domain_data],
                seq_len,
                short_term_len,
                feature_count,
                predict_length,
                device,
                rank,
                max_pairs=200
            )
            train_sampler = DistributedSampler(
                preference_pairs,
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=True
            )
            train_loader = DataLoader(
                preference_pairs,
                batch_size=batch_size,
                sampler=train_sampler,
                pin_memory=True
            )

            with higher.innerloop_ctx(
                    base_model,
                    meta_optimizer,
                    copy_initial_weights=False,
                    track_higher_grads=True,
                    device=device
            ) as (fmodel, diffopt):

                for step in range(n_inner_steps):
                    inner_loss = 0.0
                    train_sampler.set_epoch(epoch * n_inner_steps + step)

                    for i, (preferred, non_preferred) in enumerate(train_loader):
                        preferred = preferred.to(device, non_blocking=True)
                        non_preferred = non_preferred.to(device, non_blocking=True)

                        torch.cuda.empty_cache()

                        with torch.cuda.amp.autocast():
                            out_pre = fmodel(preferred)
                            out_non_pre = fmodel(non_preferred)
                            loss = dpo_loss(out_pre, out_non_pre) / accumulation_steps
                            inner_loss += loss.item()

                        if (i + 1) % accumulation_steps == 0:
                            diffopt.step(loss)
                            if hasattr(diffopt, 'zero_grad'):
                                diffopt.zero_grad()

                    inner_loss /= len(train_loader)

                # 验证阶段

                val_loss = 0.0
                val_loader = DataLoader(preference_pairs, batch_size=32, shuffle=True)
                for preferred, non_preferred in val_loader:
                    preferred, non_preferred = preferred.to(device), non_preferred.to(device)
                    out_pre = fmodel(preferred)
                    out_non_pre = fmodel(non_preferred)
                    loss = dpo_loss(out_pre, out_non_pre)
                    val_loss += loss
                val_loss /= len(val_loader)
                meta_loss += val_loss

        # 元优化步骤
        meta_loss = meta_loss / len(meta_train_data_list)
        meta_optimizer.step()
        meta_optimizer.zero_grad()

        if rank == 0:
            print(f'Epoch {epoch + 1}/{n_outer_steps}, Meta Loss: {meta_loss}')

        # 保存检查点
        if rank == 0 and (epoch + 1) % 1 == 0:
            checkpoint_path = f'model_checkpoint_epoch_{epoch + 1}_{get_timestamp()}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': meta_optimizer.state_dict(),
                'loss': meta_loss,
            }, checkpoint_path)

    return model