import torch
import higher
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from sklearn.metrics import mean_squared_error
import logging
import time

from data.dataset import InferenceDataset, split_few_shot_data
from utils.distributed import dpo_loss
from models.transformer import CustomTransformerModel
from trainers.meta_trainer import generate_preference_pairs_cross_domain


def fast_adapt_with_preferences(rank, model, target_domain_data, seq_len, short_term_len,
                                feature_count, predict_length, device, n_adapt_steps=5, lr_adapt=0.01):
    """
    Quickly adapt the meta-learned model to a new target domain using preference learning.

    Args:
        rank (int): Current process rank
        model (nn.Module): Meta-learned model
        target_domain_data (np.ndarray): Data from target domain
        seq_len (int): Length of input sequence
        short_term_len (int): Length of short-term sequence
        feature_count (int): Number of input features
        predict_length (int): Length of prediction sequence
        device (torch.device): Device to use
        n_adapt_steps (int): Number of adaptation steps
        lr_adapt (float): Learning rate for adaptation

    Returns:
        nn.Module: Adapted model
    """
    base_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.Adam(base_model.parameters(), lr=lr_adapt)

    with higher.innerloop_ctx(
            base_model,
            optimizer,
            copy_initial_weights=True,
            track_higher_grads=True
    ) as (fmodel, diffopt):

        batch_size = 32
        accumulation_steps = 3

        for step in range(n_adapt_steps):
            preference_pairs = generate_preference_pairs_cross_domain(
                fmodel,
                [target_domain_data],
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

            total_loss = 0.0
            num_batches = len(train_loader)

            for i, (preferred, non_preferred) in enumerate(train_loader):
                preferred = preferred.to(device, non_blocking=True)
                non_preferred = non_preferred.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    out_pre = fmodel(preferred)
                    out_non_pre = fmodel(non_preferred)
                    loss = dpo_loss(out_pre, out_non_pre) / accumulation_steps
                    total_loss += loss.item()

                diffopt.step(loss)

                if (i + 1) % accumulation_steps == 0:
                    torch.cuda.empty_cache()

            if rank == 0:
                logging.info(f'Adaptation step {step + 1}/{n_adapt_steps}, Loss: {total_loss / num_batches:.4f}')

        adapted_state_dict = {
            k: v.clone() for k, v in fmodel.state_dict().items()
        }

    adapted_model = CustomTransformerModel(
        input_dim=feature_count,
        hidden_dim=base_model.input_projection.out_features,  # Use same hidden dim as base model
        predict_length=predict_length,
        nhead=len(base_model.transformer_encoder.layers[0].self_attn.head_dim),
        num_layers=len(base_model.transformer_encoder.layers),
        dropout=base_model.transformer_encoder.layers[0].dropout.p,
        freeze_parameters=True
    ).to(device)

    adapted_model.load_state_dict(adapted_state_dict)

    if dist.get_world_size() > 1:
        adapted_model = DDP(
            adapted_model,
            device_ids=[rank]
        )

    return adapted_model


def evaluate_few_with_adaptation(rank, model, data, seq_len, short_term_len, feature_count,
                                 predict_length, device, model_name):
    """
    Evaluate the model on few-shot data after adaptation.

    Args:
        rank (int): Current process rank
        model (nn.Module): Model to evaluate
        data (np.ndarray): Evaluation data
        seq_len (int): Length of input sequence
        short_term_len (int): Length of short-term sequence
        feature_count (int): Number of input features
        predict_length (int): Length of prediction sequence
        device (torch.device): Device to use
        model_name (str): Name of the model for logging

    Returns:
        float or None: Mean squared error if rank is 0, None otherwise
    """
    try:
        start_time = time.time()
        train_few, test_few = split_few_shot_data(data)

        # Adapt model to new domain
        adapted_model = fast_adapt_with_preferences(
            rank,
            model,
            train_few,
            seq_len,
            short_term_len,
            feature_count,
            predict_length,
            device
        )

        # Prepare test dataset
        test_dataset = InferenceDataset(
            test_few[:, :feature_count],
            seq_len,
            predict_length
        )

        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            sampler=test_sampler,
            pin_memory=True
        )

        predictions = []
        actuals = []

        adapted_model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x_long, y in test_loader:
                x_long = x_long.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                outputs = adapted_model(x_long)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y.cpu().numpy())

                torch.cuda.empty_cache()

        # Gather results from all processes
        all_predictions = [None for _ in range(dist.get_world_size())]
        all_actuals = [None for _ in range(dist.get_world_size())]

        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_actuals, actuals)

        if rank == 0:
            # Combine predictions from all processes
            predictions = [p for pred_list in all_predictions for p in pred_list]
            actuals = [a for act_list in all_actuals for a in act_list]

            # Calculate metrics
            mse = mean_squared_error(actuals, predictions)
            duration = time.time() - start_time

            logging.info(f'{model_name} Evaluation completed in {duration:.2f}s')
            logging.info(f'{model_name} Test MSE after adaptation: {mse:.4f}')

            return mse

        return None

    except Exception as e:
        logging.error(f"Error in evaluation: {str(e)}")
        raise e