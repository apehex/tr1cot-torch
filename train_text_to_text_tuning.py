'''Fine-tuning script for Stable Diffusion on text-to-text tasks.'''

import argparse
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

# CONSTANTS ####################################################################

DATASET_NAME_MAPPING = {"lambdalabs/naruto-blip-captions": ("image", "text"),}

# VALIDATION ###################################################################

logger = accelerate.logging.get_logger(__name__, log_level="INFO")

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
        args.model_name,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers:
        pipeline.enable_xformers()

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast(accelerator.device.type):
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images

# ARGS #########################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # random config
    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help='A seed for reproducible training.')
    # output config
    parser.add_argument('--output_dir', type=str, default='outputs', required=False, help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--cache_dir', type=str, default=None, required=False, help='The directory where the downloaded models and datasets will be stored.')
    parser.add_argument('--logging_dir', type=str, default='logs', required=False, help='[TensorBoard](https://www.tensorflow.org/tensorboard) log directory.')
    parser.add_argument("--project_name", type=str, default="text-to-text", help="The `project_name` passed to the trackers https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator")
    # model config
    parser.add_argument('--model_name', type=str, default=None, required=False, help='Path to pretrained model or model identifier from huggingface.co/models.')
    parser.add_argument('--revision', type=str, default=None, required=False, help='Revision of pretrained model identifier from huggingface.co/models.')
    parser.add_argument('--variant', type=str, default=None, required=False, help='Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. fp16')
    parser.add_argument('--lora_rank', type=int, default=8, required=False, help='The dimension of the LoRA update matrices.')
    # dataset config
    parser.add_argument('--dataset_name', type=str, default='apehex/ascii-art-datacompdr-12m', required=False, help='The name of the Dataset (from the HuggingFace hub) to train on.')
    parser.add_argument('--dataset_config', type=str, default='default', required=False, help='The config of the Dataset, leave as None if there\'s only one config.')
    parser.add_argument('--dataset_split', type=str, default='train', required=False, help='The split of the Dataset.')
    parser.add_argument('--dataset_dir', type=str, default=None, required=False, help='A folder containing the training data.')
    parser.add_argument('--image_column', type=str, default='content', required=False, help='The column of the dataset containing an image.')
    parser.add_argument('--caption_column', type=str, default='caption', required=False, help='The column of the dataset containing a caption or a list of captions.')
    parser.add_argument('--max_samples', type=int, default=0, required=False, help='Truncate the number of training examples to this value if set.')
    # preprocessing config
    parser.add_argument('--resolution', type=int, default=512, required=False, help='The resolution for input images.')
    parser.add_argument('--center_crop', default=False, required=False, action='store_true', help='Whether to center (instead of random) crop the input images to the resolution.')
    parser.add_argument('--random_flip', default=False, required=False, action='store_true', help='whether to randomly flip images horizontally.')
    parser.add_argument('--image_interpolation_mode', type=str, default='lanczos', choices=[__f.lower() for __f in dir(torchvision.transforms.InterpolationMode) if not __f.startswith('__') and not __f.endswith('__')], required=False, help='The image interpolation method to use for resizing images.')
    # checkpoint config
    parser.add_argument('--resume_from', type=str, default='', required=False, help='Use a path saved by `--checkpoint_steps`, or `"latest"` to automatically select the last available checkpoint.')
    parser.add_argument('--checkpoint_steps', type=int, default=256, required=False, help='Save a checkpoint of the training state every X updates, for resuming with `--resume_from`.')
    parser.add_argument('--checkpoint_limit', type=int, default=0, required=False, help='Max number of checkpoints to store.')
    # iteration config
    parser.add_argument('--batch_dim', type=int, default=1, required=False, help='Batch size (per device) for the training dataloader.')
    parser.add_argument('--epoch_num', type=int, default=32, required=False)
    parser.add_argument('--step_num', type=int, default=0, required=False, help='Total number of training steps to perform; overrides epoch_num.')
    # learning-rate config
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=False, help='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--scale_lr', default=False, required=False, action='store_true', help='Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.')
    parser.add_argument('--lr_scheduler', type=str, default='constant', required=False, help='The scheduler type to use, among ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument('--lr_warmup_steps', type=int, default=512, required=False, help='Number of steps for the warmup in the lr scheduler.')
    # loss config
    parser.add_argument('--snr_gamma', type=float, default=0.0, required=False, help='SNR weighting gamma to rebalance the loss; ecommended value is 5.0. https://arxiv.org/pdf/2303.09556')
    # gradient config
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, required=False, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--gradient_checkpointing', default=False, required=False, action='store_true', help='Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.')
    # optimizer config
    parser.add_argument('--adam_beta1', type=float, default=0.9, required=False, help='The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, required=False, help='The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2, required=False, help='Weight decay to use.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, required=False, help='Epsilon value for the Adam optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, required=False, help='Max gradient norm.')
    # ema config
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    # precision config
    parser.add_argument('--mixed_precision', type=str, default='bf16', required=False, choices=['no', 'fp16', 'bf16'], help='Choose between fp16 and bf16 (bfloat16).')
    parser.add_argument('--allow_tf32', default=False, required=False, action='store_true', help='Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.')
    parser.add_argument('--use_8bit_adam', default=False, required=False, action='store_true', help='Whether or not to use 8-bit Adam from bitsandbytes.')
    # distribution config
    parser.add_argument('--dataloader_num_workers', type=int, default=0, required=False, help='Number of subprocesses to use for data loading; 0 means that the data will be loaded in the main process.')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)), required=False, help='For distributed training: local_rank')
    # framework config
    parser.add_argument('--enable_xformers', default=False, required=False, action='store_true', help='Whether or not to use xformers.')
    # diffusion config
    parser.add_argument('--prediction_type', type=str, default='epsilon', required=False, help='The prediction type, among "epsilon", "v_prediction" or `None`.')
    parser.add_argument('--noise_offset', type=float, default=0.0, required=False, help='The scale of noise offset.')
    parser.add_argument("--input_perturbation", type=float, default=0.0, help="The scale of input perturbation. Recommended 0.1.")
    # validation config
    parser.add_argument('--validation_prompts', type=str, default='', required=True, nargs='+', help='A set of prompts that are sampled during training for inference.')
    parser.add_argument('--validation_epochs', type=int, default=1, required=False, help='Run fine-tuning validation every X epochs.')
    # dream training
    parser.add_argument("--dream_training", default=False, required=False, action="store_true", help="Use the DREAM training method https://arxiv.org/abs/2312.00210")
    parser.add_argument("--dream_detail_preservation", type=float, default=1.0, required=False, help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)")
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    # actually process the CLI inputs
    return parser.parse_args()

# MAIN #########################################################################

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = accelerate.utils.ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='tensorboard',
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
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
    if args.seed is not None:
        accelerate.utils.set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(
        args.model_name, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = accelerate.state.AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with transformers.utils.ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = transformers.CLIPTextModel.from_pretrained(
            args.model_name, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = diffusers.AutoencoderKL.from_pretrained(
            args.model_name, subfolder="vae", revision=args.revision, variant=args.variant
        )

    unet = diffusers.UNet2DConditionModel.from_pretrained(
        args.model_name, subfolder="unet", revision=args.revision
    )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = diffusers.UNet2DConditionModel.from_pretrained(
            args.model_name, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = diffusers.training_utils.EMAModel(
            ema_unet.parameters(),
            model_cls=diffusers.UNet2DConditionModel,
            model_config=ema_unet.config,
            foreach=args.foreach_ema,
        )

    if args.enable_xformers:
        import xformers
        unet.enable_xformers()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_model = diffusers.training_utils.EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"), diffusers.UNet2DConditionModel, foreach=args.foreach_ema
            )
            ema_unet.load_state_dict(load_model.state_dict())
            if args.offload_ema:
                ema_unet.pin_memory()
            else:
                ema_unet.to(accelerator.device)
            del load_model

        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = diffusers.UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_dim * accelerator.num_processes
        )

    # Initialize the optimizer
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
        args.dataset_config,
        cache_dir=args.cache_dir,
        data_dir=args.dataset_dir)

    # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    # dataset = datasets.load_dataset(
    #     "imagefolder",
    #     data_files=os.path.join(args.dataset_dir, "**"),
    #     cache_dir=args.cache_dir)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
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
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Get the specified interpolation method from the args
    interpolation = getattr(torchvision.transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)

    # Raise an error if the interpolation method is invalid
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode {args.image_interpolation_mode}.")

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

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_samples:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_dim,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
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

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        if args.offload_ema:
            ema_unet.pin_memory()
        else:
            ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if not args.step_num:
        args.step_num = args.epoch_num * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.step_num * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.epoch_num = math.ceil(args.step_num / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if diffusers.utils.torch_utils.is_compiled_module(model) else model
        return model

    # Train!
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
                        args.dream_detail_preservation,
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
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
            args.model_name,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
            variant=args.variant,
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

            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

            for i in range(len(args.validation_prompts)):
                with torch.autocast("cuda"):
                    image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
                images.append(image)

    accelerator.end_training()


if __name__ == "__main__":
    main()
