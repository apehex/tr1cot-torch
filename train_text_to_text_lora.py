'''Fine-tuning script for Stable Diffusion with LoRA for text-to-text tasks.'''

import argparse
import contextlib
import logging
import math
import os
import pathlib
import random
import shutil

import accelerate
import accelerate.logging
import accelerate.utils
import datasets
import numpy as np
import packaging
import peft
import peft.utils
import torch
import torch.nn.functional
import torch.utils.checkpoint
import torchvision
import tqdm.auto
import transformers

import diffusers
import diffusers.optimization
import diffusers.training_utils
import diffusers.utils
import diffusers.utils.import_utils
import diffusers.utils.torch_utils

logger = accelerate.logging.get_logger(__name__, log_level='INFO')

def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f'Running validation... \n Generating {args.num_validation_images} images with prompt:'
        f' {args.validation_prompt}.'
    )
    images = []
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    generator = generator.manual_seed(args.seed)
    autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(args.num_validation_images):
            images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0])

    for tracker in accelerator.trackers:
        phase_name = 'test' if is_final_validation else 'validation'
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats='NHWC')
    return images


def parse_args():
    parser = argparse.ArgumentParser(description='Simple example of a training script.')
    # model config
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5', required=False, help='Path to pretrained model or model identifier from huggingface.co/models.')
    parser.add_argument('--revision', type=str, default=None, required=False, help='Revision of pretrained model identifier from huggingface.co/models.')
    parser.add_argument('--variant', type=str, default=None, required=False, help='Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. fp16')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, required=False, help='Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.')
    parser.add_argument('--rank', type=int, default=4, required=False, help='The dimension of the LoRA update matrices.')
    # dataset config
    parser.add_argument('--dataset_name', type=str, default='lambdalabs/naruto-blip-captions', required=False, help='The name of the Dataset (from the HuggingFace hub) to train on.')
    parser.add_argument('--dataset_config_name', type=str, default=None, required=False, help='The config of the Dataset, leave as None if there\'s only one config.')
    parser.add_argument('--train_data_dir', type=str, default=None, required=False, help='A folder containing the training data.')
    parser.add_argument('--image_column', type=str, default='image', required=False, help='The column of the dataset containing an image.')
    parser.add_argument('--caption_column', type=str, default='text', required=False, help='The column of the dataset containing a caption or a list of captions.')
    parser.add_argument('--max_train_samples', type=int, default=None, required=False, help='Truncate the number of training examples to this value if set.')
    # validation config
    parser.add_argument('--validation_prompt', type=str, default=None, required=False, help='A prompt that is sampled during training for inference.')
    parser.add_argument('--num_validation_images', type=int, default=4, required=False, help='Number of images that should be generated during validation with `validation_prompt`.')
    parser.add_argument('--validation_epochs', type=int, default=1, required=False, help='Run fine-tuning validation every X epochs.')
    # output config
    parser.add_argument('--output_dir', type=str, default='lora-model', required=False, help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--cache_dir', type=str, default=None, required=False, help='The directory where the downloaded models and datasets will be stored.')
    parser.add_argument('--logging_dir', type=str, default='logs', required=False, help='[TensorBoard](https://www.tensorflow.org/tensorboard) log directory.')
    # generic config
    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help='A seed for reproducible training.')
    # preprocessing config
    parser.add_argument('--resolution', type=int, default=512, required=False, help='The resolution for input images.')
    parser.add_argument('--center_crop', type=bool, default=False, required=False, action='store_true', help='Whether to center (instead of random) crop the input images to the resolution.')
    parser.add_argument('--random_flip', type=bool, default=True, required=False, action='store_true', help='whether to randomly flip images horizontally.')
    parser.add_argument('--image_interpolation_mode', type=str, default='lanczos', choices=[f.lower() for f in dir(torchvision.transforms.InterpolationMode) if not f.startswith('__') and not f.endswith('__')], required=False, help='The image interpolation method to use for resizing images.')
    # iteration config
    parser.add_argument('--train_batch_size', type=int, default=16, required=False, help='Batch size (per device) for the training dataloader.')
    parser.add_argument('--num_train_epochs', type=int, default=100, required=False)
    parser.add_argument('--max_train_steps', type=int, default=None, required=False, help='Total number of training steps to perform; overrides num_train_epochs.')
    parser.add_argument('--checkpointing_steps', type=int, default=256, required=False, help='Save a checkpoint of the training state every X updates, for resuming with `--resume_from_checkpoint`.')
    parser.add_argument('--checkpoints_total_limit', type=int, default=None, required=False, help='Max number of checkpoints to store.')
    # gradient config
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, required=False, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--gradient_checkpointing', type=bool, default=False, required=False, action='store_true', help='Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.')
    # learning-rate config
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=False, help='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--scale_lr', type=bool, default=False, required=False, action='store_true', help='Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.')
    parser.add_argument('--lr_scheduler', type=str, default='constant', required=False, help='The scheduler type to use, among ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument('--lr_warmup_steps', type=int, default=500, required=False, help='Number of steps for the warmup in the lr scheduler.')
    # loss config
    parser.add_argument('--snr_gamma', type=float, default=None, required=False, help='SNR weighting gamma to rebalance the loss; ecommended value is 5.0. https://arxiv.org/pdf/2303.09556')
    # precision config
    parser.add_argument('--mixed_precision', type=str, default=None, required=False, choices=['no', 'fp16', 'bf16'], help='Choose between fp16 and bf16 (bfloat16).')
    parser.add_argument('--allow_tf32', type=bool, default=False, required=False, action='store_true', help='Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.')
    parser.add_argument('--use_8bit_adam', type=bool, default=False, required=False, action='store_true', help='Whether or not to use 8-bit Adam from bitsandbytes.')
    # distribution config
    parser.add_argument('--dataloader_num_workers', type=int, default=0, required=False, help='Number of subprocesses to use for data loading; 0 means that the data will be loaded in the main process.')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)), required=False, help='For distributed training: local_rank')
    # optimizer config
    parser.add_argument('--adam_beta1', type=float, default=0.9, required=False, help='The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, required=False, help='The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2, required=False, help='Weight decay to use.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, required=False, help='Epsilon value for the Adam optimizer')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='Max gradient norm.')
    # implementation config
    parser.add_argument('--enable_xformers_memory_efficient_attention', type=bool, default=False, required=False, action='store_true', help='Whether or not to use xformers.')
    # other config
    parser.add_argument('--prediction_type', type=str, default=None, required=False, help='The prediction type, among "epsilon", "v_prediction" or `None`.')
    parser.add_argument('--noise_offset', type=float, default=0, required=False, help='The scale of noise offset.')

    args = parser.parse_args()

    return args


DATASET_NAME_MAPPING = {
    'lambdalabs/naruto-blip-captions': ('image', 'text'),
}


def main():
    args = parse_args()

    logging_dir = pathlib.Path(args.output_dir, args.logging_dir)

    accelerator_project_config = accelerate.utils.ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir)

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='tensorboard',
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    accelerate.utils.set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='tokenizer', revision=args.revision
    )
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='text_encoder', revision=args.revision
    )
    vae = diffusers.AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='vae', revision=args.revision, variant=args.variant
    )
    unet = diffusers.UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='unet', revision=args.revision, variant=args.variant
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    unet_lora_config = peft.LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights='gaussian',
        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == 'fp16':
        # only upcast trainable parameters (LoRA) into fp32
        diffusers.training_utils.cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if diffusers.utils.import_utils.is_xformers_available():
            import xformers

            xformers_version = packaging.version.parse(xformers.__version__)
            if xformers_version == packaging.version.parse('0.0.16'):
                logger.warning(
                    'xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.'
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError('xformers is not available. Make sure it is installed correctly')

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes
        except ImportError:
            raise ImportError(
                'Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`'
            )

        optimizer_cls = bitsandbytes.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)

    # download the dataset.
    dataset = datasets.load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        data_dir=args.train_data_dir,)

    # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    # dataset = datasets.load_dataset(
    #     'imagefolder',
    #     data_files={'train': os.path.join(args.train_data_dir, '**')},
    #     cache_dir=args.cache_dir,)

    # tokenize inputs and targets.
    column_names = dataset['train'].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f'--image_column" value "{args.image_column}" needs to be one of: {", ".join(column_names)}'
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f'--caption_column" value "{args.caption_column}" needs to be one of: {", ".join(column_names)}'
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f'Caption column `{caption_column}` should contain either strings or lists of strings.'
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        return inputs.input_ids

    # Get the specified interpolation method from the args
    interpolation = getattr(torchvision.transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)

    # Raise an error if the interpolation method is invalid
    if interpolation is None:
        raise ValueError(f'Unsupported interpolation mode {args.image_interpolation_mode}.')

    # Data preprocessing transformations
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.resolution, interpolation=interpolation),  # Use dynamic interpolation method
            torchvision.transforms.CenterCrop(args.resolution) if args.center_crop else torchvision.transforms.RandomCrop(args.resolution),
            torchvision.transforms.RandomHorizontalFlip() if args.random_flip else torchvision.transforms.Lambda(lambda x: x),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if diffusers.utils.torch_utils.is_compiled_module(model) else model
        return model

    def preprocess_train(examples):
        images = [image.convert('RGB') for image in examples[image_column]]
        examples['pixel_values'] = [train_transforms(image) for image in images]
        examples['input_ids'] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset['train'].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example['input_ids'] for example in examples])
        return {'pixel_values': pixel_values, 'input_ids': input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f'The length of the "train_dataloader" after "accelerator.prepare" ({len(train_dataloader)}) does not match '
                f'the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. '
                f'This inconsistency may result in the learning rate scheduler not functioning properly.'
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers('text2image-fine-tune', config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != 'latest':
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint')]
            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f'Checkpoint "{args.resume_from_checkpoint}" does not exist. Starting a new training run.'
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f'Resuming from checkpoint {path}')
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split('-')[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm.auto.tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc='Steps',
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch['pixel_values'].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch['input_ids'], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f'Unknown prediction type {noise_scheduler.config.prediction_type}')

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction='mean')
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = diffusers.training_utils.compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == 'epsilon':
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == 'v_prediction':
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction='none')
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({'train_loss': train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f'{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints'
                                )
                                logger.info(f'removing checkpoints: {", ".join(removing_checkpoints)}')

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = diffusers.utils.convert_state_dict_to_diffusers(
                            peft.utils.get_peft_model_state_dict(unwrapped_unet)
                        )

                        diffusers.StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f'Saved state to {save_path}')

            logs = {'step_loss': loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = diffusers.DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                images = log_validation(pipeline, args, accelerator, epoch)

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = diffusers.utils.convert_state_dict_to_diffusers(peft.utils.get_peft_model_state_dict(unwrapped_unet))
        diffusers.StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        # Final inference
        # Load previous pipeline
        if args.validation_prompt is not None:
            pipeline = diffusers.DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )

            # load attention processors
            pipeline.load_lora_weights(args.output_dir)

            # run inference
            images = log_validation(pipeline, args, accelerator, epoch, is_final_validation=True)

    accelerator.end_training()


if __name__ == '__main__':
    main()
