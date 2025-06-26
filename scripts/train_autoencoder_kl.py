import argparse
import contextlib
import gc
import logging
import math
import os
import shutil
import pathlib

import datasets
import lpips
import numpy
import torch
import torch.nn.functional
import torch.utils.checkpoint
import torchvision
import transformers

import accelerate
import accelerate.logging
import accelerate.utils
import packaging
import PIL as pillow
import taming.modules.losses.vqperceptual
import torchvision
import tqdm.auto

import diffusers
import diffusers.optimization
import diffusers.training_utils
import diffusers.utils.torch_utils

import mlable.meta

# VALIDATION ###################################################################

logger = accelerate.logging.get_logger(__name__)

@torch.no_grad()
def log_validation(vae, args, accelerator, weight_dtype, step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation:
        vae = accelerator.unwrap_model(vae)
    else:
        vae = diffusers.AutoencoderKL.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    images = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

    __transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.image_resolution, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(args.image_resolution),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )

    for i, validation_image in enumerate(args.validation_images):
        validation_image = pillow.Image.open(validation_image).convert("RGB")
        targets = __transforms(validation_image).to(accelerator.device, weight_dtype)
        targets = targets.unsqueeze(0)

        with inference_ctx:
            reconstructions = vae(targets).sample

        images.append(torch.cat([targets.cpu(), reconstructions.cpu()], axis=0))

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = numpy.stack([numpy.asarray(img) for img in images])
            tracker.writer.add_images(f"{tracker_key}: Original (left), Reconstruction (right)", np_images, step)
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        gc.collect()
        torch.cuda.empty_cache()

    return images

# MODEL ########################################################################

def unwrap_model(accelerator, model):
    __model = accelerator.unwrap_model(model)
    return __model._orig_mod if diffusers.utils.torch_utils.is_compiled_module(__model) else __model

# DATASET ######################################################################

def make_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config,
            cache_dir=args.cache_dir,
            data_dir=args.dataset_dir,
        )
    else:
        data_files = {}
        if args.dataset_dir is not None:
            data_files["train"] = os.path.join(args.dataset_dir, "**")
        dataset = datasets.load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    __transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.image_resolution, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(args.image_resolution),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [__transforms(image) for image in images]

        examples["pixel_values"] = images

        return examples

    with accelerator.main_process_first():
        if args.max_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.random_seed).select(range(args.max_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {"pixel_values": pixel_values}


def main():
    args = mlable.meta.parse_args(
        definitions=mlable.meta.VAE_ARGS,
        description='')

    accelerator_project_config = accelerate.utils.ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.logging_dir)

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='tensorboard',
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # set the training seed now.
    accelerate.utils.set_seed(args.random_seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load AutoencoderKL
    if args.model_name is None and args.model_config is None:
        config = diffusers.AutoencoderKL.load_config("stabilityai/sd-vae-ft-mse")
        vae = diffusers.AutoencoderKL.from_config(config)
    elif args.model_name is not None:
        vae = diffusers.AutoencoderKL.from_pretrained(args.model_name, revision=args.model_revision)
    else:
        config = diffusers.AutoencoderKL.load_config(args.model_config)
        vae = diffusers.AutoencoderKL.from_config(config)
    if args.use_ema:
        ema_vae = diffusers.training_utils.EMAModel(vae.parameters(), model_cls=diffusers.AutoencoderKL, model_config=vae.config)
    perceptual_loss = lpips.LPIPS(net="vgg").eval()
    discriminator = taming.modules.losses.vqperceptual.NLayerDiscriminator(input_nc=3, n_layers=3, use_actnorm=False).apply(taming.modules.losses.vqperceptual.weights_init)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    # `accelerate` 0.16.0 will have better support for customized saving
    if packaging.version.parse(accelerate.__version__) >= packaging.version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    sub_dir = "autoencoderkl_ema"
                    ema_vae.save_pretrained(os.path.join(output_dir, sub_dir))

                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    if isinstance(model, diffusers.AutoencoderKL):
                        sub_dir = "autoencoderkl"
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                    else:
                        sub_dir = "discriminator"
                        os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(output_dir, sub_dir, "pytorch_model.bin"))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                if args.use_ema:
                    sub_dir = "autoencoderkl_ema"
                    load_model = diffusers.training_utils.EMAModel.from_pretrained(os.path.join(input_dir, sub_dir), diffusers.AutoencoderKL)
                    ema_vae.load_state_dict(load_model.state_dict())
                    ema_vae.to(accelerator.device)
                    del load_model

                # pop models so that they are not loaded again
                model = models.pop()
                load_model = taming.modules.losses.vqperceptual.NLayerDiscriminator(input_nc=3, n_layers=3, use_actnorm=False).load_state_dict(
                    os.path.join(input_dir, "discriminator", "pytorch_model.bin")
                )
                model.load_state_dict(load_model.state_dict())
                del load_model

                model = models.pop()
                load_model = diffusers.AutoencoderKL.from_pretrained(input_dir, subfolder="autoencoderkl")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(True)
    vae.train()
    discriminator.requires_grad_(True)
    discriminator.train()

    if args.enable_xformers:
        import xformers
        vae.enable_xformers()

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(vae).dtype != torch.float32:
        raise ValueError(f"VAE loaded as datatype {unwrap_model(vae).dtype}. {low_precision_error_string}")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_dim * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = filter(lambda p: p.requires_grad, vae.parameters())
    disc_params_to_optimize = filter(lambda p: p.requires_grad, discriminator.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    disc_optimizer = optimizer_class(
        disc_params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = make_train_dataset(args, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_dim,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_step_num = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.step_num is None:
        args.step_num = args.epoch_num * num_update_steps_per_epoch
        overrode_step_num = True

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.step_num * accelerator.num_processes,
        num_cycles=args.lr_cycle_num,
        power=args.lr_power,
    )
    disc_lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=disc_optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.step_num * accelerator.num_processes,
        num_cycles=args.lr_cycle_num,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    (
        vae,
        discriminator,
        optimizer,
        disc_optimizer,
        train_dataloader,
        lr_scheduler,
        disc_lr_scheduler,
    ) = accelerator.prepare(
        vae, discriminator, optimizer, disc_optimizer, train_dataloader, lr_scheduler, disc_lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move VAE, perceptual loss and discriminator to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    perceptual_loss.to(accelerator.device, dtype=weight_dtype)
    discriminator.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_step_num:
        args.step_num = args.epoch_num * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.epoch_num = math.ceil(args.step_num / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        __config = dict(vars(args))
        __config.pop('validation_prompts')
        __config.pop('validation_images')
        accelerator.init_trackers(args.project_name, __config)

    # Train!
    total_batch_size = args.batch_dim * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.epoch_num}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_dim}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.step_num}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from:
        if args.resume_from != "latest":
            path = os.path.basename(args.resume_from)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from}' does not exist. Starting a new training run."
            )
            args.resume_from = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm.auto.tqdm(
        range(0, args.step_num),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.epoch_num):
        vae.train()
        discriminator.train()
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space and reconstruct from them
            targets = batch["pixel_values"].to(dtype=weight_dtype)
            posterior = accelerator.unwrap_model(vae).encode(targets).latent_dist
            latents = posterior.sample()
            reconstructions = accelerator.unwrap_model(vae).decode(latents).sample

            if (step // args.gradient_accumulation_steps) % 2 == 0 or global_step < args.disc_start:
                with accelerator.accumulate(vae):
                    # reconstruction loss. Pixel level differences between input vs output
                    if args.rec_loss == "l2":
                        rec_loss = torch.nn.functional.mse_loss(reconstructions.float(), targets.float(), reduction="none")
                    elif args.rec_loss == "l1":
                        rec_loss = torch.nn.functional.l1_loss(reconstructions.float(), targets.float(), reduction="none")
                    else:
                        raise ValueError(f"Invalid reconstruction loss type: {args.rec_loss}")
                    # perceptual loss. The high level feature mean squared error loss
                    with torch.no_grad():
                        p_loss = perceptual_loss(reconstructions, targets)

                    rec_loss = rec_loss + args.lpips_rate * p_loss
                    nll_loss = rec_loss
                    nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

                    kl_loss = posterior.kl()
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                    logits_fake = discriminator(reconstructions)
                    g_loss = -torch.mean(logits_fake)
                    last_layer = accelerator.unwrap_model(vae).decoder.conv_out.weight
                    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
                    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
                    disc_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
                    disc_weight = torch.clamp(disc_weight, 0.0, 1e4).detach()
                    disc_weight = disc_weight * args.disc_scale
                    disc_factor = args.disc_factor if global_step >= args.disc_start else 0.0

                    loss = nll_loss + args.kl_rate * kl_loss + disc_weight * disc_factor * g_loss

                    logs = {
                        "loss": loss.detach().mean().item(),
                        "nll_loss": nll_loss.detach().mean().item(),
                        "rec_loss": rec_loss.detach().mean().item(),
                        "p_loss": p_loss.detach().mean().item(),
                        "kl_loss": kl_loss.detach().mean().item(),
                        "disc_weight": disc_weight.detach().mean().item(),
                        "disc_factor": disc_factor,
                        "g_loss": g_loss.detach().mean().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = vae.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=False)
            else:
                with accelerator.accumulate(discriminator):
                    logits_real = discriminator(targets)
                    logits_fake = discriminator(reconstructions)
                    disc_loss = taming.modules.losses.vqperceptual.hinge_d_loss if args.disc_loss == "hinge" else taming.modules.losses.vqperceptual.vanilla_d_loss
                    disc_factor = args.disc_factor if global_step >= args.disc_start else 0.0
                    d_loss = disc_factor * disc_loss(logits_real, logits_fake)
                    logs = {
                        "disc_loss": d_loss.detach().mean().item(),
                        "logits_real": logits_real.detach().mean().item(),
                        "logits_fake": logits_fake.detach().mean().item(),
                        "disc_lr": disc_lr_scheduler.get_last_lr()[0],
                    }
                    accelerator.backward(d_loss)
                    if accelerator.sync_gradients:
                        params_to_clip = discriminator.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    disc_optimizer.step()
                    disc_lr_scheduler.step()
                    disc_optimizer.zero_grad(set_to_none=False)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if args.use_ema:
                    ema_vae.step(vae.parameters())

                if accelerator.is_main_process:
                    if global_step % args.checkpoint_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoint_limit`
                        if args.checkpoint_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoint_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoint_limit:
                                num_to_remove = len(checkpoints) - args.checkpoint_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step == 1 or global_step % args.validation_steps == 0:
                        if args.use_ema:
                            ema_vae.store(vae.parameters())
                            ema_vae.copy_to(vae.parameters())
                        image_logs = log_validation(
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )
                        if args.use_ema:
                            ema_vae.restore(vae.parameters())

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.step_num:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        discriminator = accelerator.unwrap_model(discriminator)
        if args.use_ema:
            ema_vae.copy_to(vae.parameters())
        vae.save_pretrained(args.output_dir)
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        # Run a final round of validation.
        image_logs = None
        image_logs = log_validation(
            vae=vae,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            step=global_step,
            is_final_validation=True,
        )

    accelerator.end_training()

if __name__ == "__main__":
    main()
