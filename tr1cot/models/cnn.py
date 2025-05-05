import functools

import tensorflow as tf

import mlable.layers.embedding
import mlable.models.diffusion

import tr1cot.layers.convolution

# CONSTANTS ####################################################################

START_RATE = 0.95 # signal rate at the start of the forward diffusion process
END_RATE = 0.02 # signal rate at the end of the forward diffusion process

# END-TO-END ###################################################################

@tf.keras.utils.register_keras_serializable(package='models')
class UnetDiffusionModel(mlable.models.diffusion.LatentDiffusionModel):
    def __init__(self, block_num: int, latent_dim: iter, start_rate: float=START_RATE, end_rate: float=END_RATE, **kwargs) -> None:
        super(UnetDiffusionModel, self).__init__(**kwargs)
        # save the config to init later
        self._config.update({'block_num': block_num, 'latent_dim': [latent_dim] if isinstance(latent_dim, int) else list(latent_dim),})
        # layers
        self._match_block = None
        self._concat_block = None
        self._embed_height_block = None
        self._embed_width_block = None
        self._expand_block = None
        self._encode_blocks = []
        self._transform_blocks = []
        self._decode_blocks = []
        self._project_block = None
        # custom build
        self._built = False

    def build(self, input_shape: tuple) -> None:
        # unpack
        __shape_o = tuple(input_shape[0])
        __shape_v = tuple(input_shape[-1])
        # build the core diffusion model
        super(UnetDiffusionModel, self).build(__shape_o)
        # init
        self._match_block = tf.keras.layers.UpSampling2D(size=__shape_o[1:3], interpolation="nearest")
        self._concat_block = tf.keras.layers.Concatenate(axis=-1)
        self._embed_height_block = mlable.layers.embedding.PositionalEmbedding(sequence_axis=1, feature_axis=-1)
        self._embed_width_block = mlable.layers.embedding.PositionalEmbedding(sequence_axis=2, feature_axis=-1)
        self._expand_block = tf.keras.layers.Dense(units=self._config['latent_dim'][0], activation=None, use_bias=True)
        self._encode_blocks = [tr1cot.layers.convolution.DownBlock(block_dim=__d, block_num=self._config['block_num']) for __d in self._config['latent_dim'][:-1]]
        self._transform_blocks = [tr1cot.layers.convolution.ResidualBlock(latent_dim=self._config['latent_dim'][-1]) for _ in range(self._config['block_num'])]
        self._decode_blocks = [tr1cot.layers.convolution.UpBlock(block_dim=__d, block_num=self._config['block_num']) for __d in reversed(self._config['latent_dim'][:-1])]
        self._project_block = tf.keras.layers.Dense(units=__shape_o[-1], kernel_initializer='zeros')
        # build
        self._match_block.build(__shape_v)
        __shape_v = self._match_block.compute_output_shape(__shape_v)
        self._concat_block.build([__shape_o, __shape_v])
        __shape_o = self._concat_block.compute_output_shape([__shape_o, __shape_v])
        self._embed_height_block.build(__shape_o)
        __shape_o = self._embed_height_block.compute_output_shape(__shape_o)
        self._embed_width_block.build(__shape_o)
        __shape_o = self._embed_width_block.compute_output_shape(__shape_o)
        self._expand_block.build(__shape_o)
        __shape_o = self._expand_block.compute_output_shape(__shape_o)
        for __b in self._encode_blocks:
            __b.build(__shape_o)
            __shape_o = __b.compute_output_shape(__shape_o)
        for __b in self._transform_blocks:
            __b.build(__shape_o)
            __shape_o = __b.compute_output_shape(__shape_o)
        for __b in self._decode_blocks:
            __b.build(__shape_o)
            __shape_o = __b.compute_output_shape(__shape_o)
        self._project_block.build(__shape_o)
        __shape_o = self._project_block.compute_output_shape(__shape_o)
        # register
        self.built = True

    def call(self, inputs: tuple, training: bool=False, **kwargs) -> tf.Tensor:
        __dtype = self.compute_dtype
        # unpack
        __outputs, __variances = tf.cast(inputs[0], dtype=__dtype), tf.cast(inputs[-1], dtype=__dtype)
        # match (B, 1, 1, 1) => (B, H, W, 1)
        __variances = self._match_block(__variances)
        # merge (B, H, W, E) + (B, H, W, 1) => (B, H, W, E+1)
        __outputs = self._concat_block([__outputs, __variances])
        # embed the spatial axes (B, H, W, E+1)
        __outputs = self._embed_height_block(__outputs)
        __outputs = self._embed_width_block(__outputs)
        # expand (B, H, W, E+1) => (B, H, W, L)
        __outputs = self._expand_block(__outputs)
        # save residuals that skip the whole sampling process
        __residuals = __outputs
        # downsample (B, Hi, Wi, Li) => (B, Hi/2, Wi/2, Li+1)
        __outputs = functools.reduce(lambda __x, __b: __b(__x, training=training), self._encode_blocks, __outputs)
        # transform (B, Hn, Wn, Ln) => (B, Hn, Wn, Ln)
        __outputs = functools.reduce(lambda __x, __b: __b(__x, training=training), self._transform_blocks, __outputs)
        # upsample (B, Hi, Wi, Li) => (B, 2Hi, 2Wi, Li-1)
        __outputs = functools.reduce(lambda __x, __b: __b(__x, training=training), self._decode_blocks, __outputs)
        # project (B, H, W, L) => (B, H, W, 1)
        return self._project_block(__outputs + __residuals)

    def get_config(self) -> dict:
        __config = super(UnetDiffusionModel, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
