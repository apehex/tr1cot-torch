'''Fine-tuning script for Stable Diffusion on text-to-text tasks.'''

import argparse
import functools
import logging
import math
import os
import random
import shutil

import datasets
import numpy as np
import tqdm.auto

import torch
import torch.nn.functional
import torch.utils.checkpoint
import torchvision

import transformers
import transformers.utils

import diffusers
import diffusers.optimization
import diffusers.training_utils
import diffusers.utils.torch_utils

import accelerate
import accelerate.logging
import accelerate.state
import accelerate.utils

import mlable.meta

# CONSTANTS ####################################################################

DATASET_NAME_MAPPING = {'lambdalabs/naruto-blip-captions': ('image', 'text'),}

# VALIDATION ###################################################################

logger = accelerate.logging.get_logger(__name__, log_level='INFO')

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info('Running validation... ')

    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
        args.model_name,
        vae=vae, # accelerator.unwrap_model(vae),
        text_encoder=text_encoder, # accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.model_revision,
        variant=args.model_variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers:
        pipeline.enable_xformers()

    generator = torch.Generator(device=accelerator.device).manual_seed(args.random_seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast(accelerator.device.type):
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images('validation', np_images, epoch, dataformats='NHWC')
        else:
            logger.warning(f'image logging not implemented for {tracker.name}')

    del pipeline
    torch.cuda.empty_cache()

    return images

# MODEL ########################################################################

def unwrap_model(accelerator, model):
    __model = accelerator.unwrap_model(model)
    return __model._orig_mod if diffusers.utils.torch_utils.is_compiled_module(__model) else __model

# DEEPSPEED ####################################################################

def deepspeed_zero_init_disabled_context_manager():
    deepspeed_plugins = [accelerate.state.AcceleratorState().deepspeed_plugin] if accelerate.state.is_initialized() else []
    # disable zero.Init
    return [__p.zero3_init_context_manager(enable=False) for __p in deepspeed_plugins if __p is not None]

# IO HOOKS #####################################################################

def save_model_hook(models, weights, output_dir, ema_model, accelerator, args):
    if accelerator.is_main_process:
        if args.use_ema:
            ema_model.save_pretrained(os.path.join(output_dir, 'unet_ema'))

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, 'unet'))
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

def load_model_hook(models, input_dir, ema_model, accelerator, args):
    if args.use_ema:
        load_model = diffusers.training_utils.EMAModel.from_pretrained(
            os.path.join(input_dir, 'unet_ema'), diffusers.UNet2DConditionModel, foreach=args.foreach_ema
        )
        ema_model.load_state_dict(load_model.state_dict())
        if args.offload_ema:
            ema_model.pin_memory()
        else:
            ema_model.to(accelerator.device)
        # cleanup
        del load_model

    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()
        # load diffusers style into model
        load_model = diffusers.UNet2DConditionModel.from_pretrained(input_dir, subfolder='unet')
        model.register_to_config(**load_model.config)
        model.load_state_dict(load_model.state_dict())
        # cleanup
        del load_model

# IMAGES #######################################################################

def preprocess_images(examples: dict, transforms: callable, image_column: str='image') -> list:
    return [transforms(__i.convert("RGB")) for __i in examples[image_column]]

# CAPTIONS #####################################################################

def tokenize_captions(captions, tokenizer):
    return tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt").input_ids

def preprocess_captions(examples: dict, tokenizer: callable, caption_column: str='text') -> list:
    __captions = [(__c if isinstance(__c, str) else random.choice(__c)) for __c in examples[caption_column]]
    return tokenize_captions(captions=__captions, tokenizer=tokenizer)

# PREPROCESSING ################################################################

def preprocess(examples: dict, transforms: callable, tokenizer: callable, image_column: str='image', caption_column: str='text') -> dict:
    examples["pixel_values"] = preprocess_images(examples, transforms=transforms, image_column=image_column)
    examples["input_ids"] = preprocess_captions(examples, tokenizer=tokenizer, caption_column=caption_column)
    return examples

def collate_fn(examples: iter):
    __input_ids = torch.stack([__e['input_ids'] for __e in examples])
    __pixel_values = torch.stack([__e['pixel_values'] for __e in examples])
    __pixel_values = __pixel_values.to(memory_format=torch.contiguous_format).float()
    return {'pixel_values': __pixel_values, 'input_ids': __input_ids}

# MAIN #########################################################################

def main():
    args = mlable.meta.parse_args(
        definitions=mlable.meta.DDPM_ARGS,
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

    # make one log on every process with the configuration for debugging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,)
    logger.info(accelerator.state, main_process_only=False)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # set the training seed now
    accelerate.utils.set_seed(args.random_seed)

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(args.model_name, subfolder='scheduler')
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        args.model_name, subfolder='tokenizer', revision=args.model_revision
    )

    # exclude the 2 frozen models from partitioning during `zero.Init`
    with transformers.utils.ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = transformers.CLIPTextModel.from_pretrained(
            args.model_name, subfolder='text_encoder', revision=args.model_revision, variant=args.model_variant
        )
        vae = diffusers.AutoencoderKL.from_pretrained(
            args.model_name, subfolder='vae', revision=args.model_revision, variant=args.model_variant
        )

    unet = diffusers.UNet2DConditionModel.from_pretrained(
        args.model_name, subfolder='unet', revision=args.model_revision
    )

    # freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # create EMA for the unet
    if args.use_ema:
        ema_unet = diffusers.UNet2DConditionModel.from_pretrained(
            args.model_name, subfolder='unet', revision=args.model_revision, variant=args.model_variant
        )
        ema_unet = diffusers.training_utils.EMAModel(
            ema_unet.parameters(),
            model_cls=diffusers.UNet2DConditionModel,
            model_config=ema_unet.config,
            foreach=args.foreach_ema,
        )
    else:
        ema_unet = None

    # xformer attention
    if args.enable_xformers:
        import xformers
        unet.enable_xformers()

    accelerator.register_save_state_pre_hook(functools.partial(save_model_hook, ema_model=ema_unet, accelerator=accelerator, args=args))
    accelerator.register_load_state_pre_hook(functools.partial(load_model_hook, ema_model=ema_unet, accelerator=accelerator, args=args))

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_dim * accelerator.num_processes
        )

    # init the optimizer
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # download the dataset
    dataset = datasets.load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        data_dir=args.dataset_dir)

    # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    # dataset = datasets.load_dataset(
    #     "imagefolder",
    #     data_files=os.path.join(args.dataset_dir, "**"),
    #     cache_dir=args.cache_dir)

    # select the fields in the dataset to parse data from
    column_names = dataset.column_names
    image_column = args.image_column if (args.image_column in column_names) else column_names[0]
    caption_column = args.caption_column if (args.caption_column in column_names) else column_names[1]

    # get the interpolation method from the args
    interpolation = getattr(torchvision.transforms.InterpolationMode, args.interpolation_mode.upper(), 'lanczos')

    # image transformations
    train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.image_resolution, interpolation=interpolation),  # Use dynamic interpolation method
            torchvision.transforms.CenterCrop(args.image_resolution) if args.center_crop else torchvision.transforms.RandomCrop(args.image_resolution),
            torchvision.transforms.RandomHorizontalFlip() if args.random_flip else torchvision.transforms.Lambda(lambda x: x),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),])

    __preprocess = functools.partial(
        preprocess,
        transforms=train_transforms,
        tokenizer=tokenizer,
        image_column=image_column,
        caption_column=caption_column)

    with accelerator.main_process_first():
        if args.max_samples:
            dataset = dataset.shuffle(seed=args.random_seed).select(range(args.max_samples))
        # Set the training transforms
        train_dataset = dataset.with_transform(__preprocess)

    # distribute data loading
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_dim,
        num_workers=args.dataloader_num_workers,
    )

    # cf https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if not args.step_num:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.epoch_num * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.step_num * accelerator.num_processes

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # distribute everything across devices
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        if args.offload_ema:
            ema_unet.pin_memory()
        else:
            ema_unet.to(accelerator.device)

    # cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # match the total training steps with the dataloader size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if not args.step_num:
        args.step_num = args.epoch_num * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.step_num * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # deduce the number of epochs
    args.epoch_num = math.ceil(args.step_num / num_update_steps_per_epoch)

    # init the trackers
    if accelerator.is_main_process:
        __config = dict(vars(args))
        __config.pop('validation_prompts')
        __config.pop('validation_images')
        accelerator.init_trackers(args.project_name, __config)

    # train!
    total_batch_size = args.batch_dim * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
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

    for epoch in range(first_epoch, args.epoch_num):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.dream_training:
                    noisy_latents, target = diffusers.training_utils.compute_dream_and_update_latents(
                        unet,
                        noise_scheduler,
                        timesteps,
                        noise,
                        noisy_latents,
                        target,
                        encoder_hidden_states,
                        args.preservation_rate,
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if not args.snr_gamma:
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = diffusers.training_utils.compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.batch_dim)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    if args.offload_ema:
                        ema_unet.to(device="cuda", non_blocking=True)
                    ema_unet.step(unet.parameters())
                    if args.offload_ema:
                        ema_unet.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpoint_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoint_limit`
                        if args.checkpoint_limit:
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

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.step_num:
                break

        if accelerator.is_main_process:
            if args.validation_prompts and epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(accelerator, unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
            args.model_name,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.model_revision,
            variant=args.model_variant,
        )
        pipeline.save_pretrained(args.output_dir)

        # Run a final round of inference.
        images = []
        if args.validation_prompts:
            logger.info("Running inference for collecting generated images...")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            if args.enable_xformers:
                pipeline.enable_xformers()

            if args.random_seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.random_seed)

            for i in range(len(args.validation_prompts)):
                with torch.autocast("cuda"):
                    image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
                images.append(image)

    accelerator.end_training()


if __name__ == "__main__":
    main()
