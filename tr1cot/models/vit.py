import functools

import tensorflow as tf

import mlable.layers.embedding
import mlable.models.diffusion

import tr1cot.layers.transformer

# CONSTANTS ####################################################################

START_RATE = 0.95 # signal rate at the start of the forward diffusion process
END_RATE = 0.02 # signal rate at the end of the forward diffusion process

EPSILON = 1e-5
DROPOUT = 0.0

# END-TO-END ###################################################################

@tf.keras.utils.register_keras_serializable(package='models')
class VitDiffusionModel(mlable.models.diffusion.LatentDiffusionModel):
    def __init__(
        self,
        patch_dim: iter,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        start_rate: float=START_RATE,
        end_rate: float=END_RATE,
        **kwargs
    ) -> None:
        super(VitDiffusionModel, self).__init__(start_rate=start_rate, end_rate=end_rate, **kwargs)
        # save the config to init later
        self._config.update({
            'patch_dim': [patch_dim] if isinstance(patch_dim, int) else list(patch_dim),
            'dropout_rate': dropout_rate,
            'epsilon_rate': epsilon_rate,})
        # layers
        self._match_block = None
        self._concat_block = None
        self._embed_height_block = None
        self._embed_width_block = None
        self._even_block = None
        self._transform_blocks = []
        self._project_block = None
        # custom build
        self._built = False

    def build(self, input_shape: tuple) -> None:
        # unpack
        __shape_o = tuple(input_shape[0])
        __shape_v = tuple(input_shape[-1])
        # build the core diffusion model
        super(VitDiffusionModel, self).build(__shape_o)
        # init
        self._match_block = tf.keras.layers.UpSampling2D(size=__shape_o[1:3], interpolation="nearest")
        self._concat_block = tf.keras.layers.Concatenate(axis=-1)
        self._embed_height_block = mlable.layers.embedding.PositionalEmbedding(sequence_axis=1, feature_axis=-1)
        self._embed_width_block = mlable.layers.embedding.PositionalEmbedding(sequence_axis=2, feature_axis=-1)
        self._even_block = tf.keras.layers.Dense(units=__shape_o[-1], activation=None, use_bias=True)
        self._transform_blocks = [tr1cot.layers.transformer.PatchTransformerBlock(patch_dim=__p, dropout_rate=self._config['dropout_rate'], epsilon_rate=self._config['epsilon_rate']) for __p in self._config['patch_dim']]
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
        self._even_block.build(__shape_o)
        __shape_o = self._even_block.compute_output_shape(__shape_o)
        for __b in self._transform_blocks:
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
        # even (B, H, W, E+1) => (B, H, W, E)
        __outputs = self._even_block(__outputs)
        # transform (B, H, W, E) => (B, H, W, E)
        __outputs = functools.reduce(lambda __x, __b: __b(__x, training=training), self._transform_blocks, __outputs)
        # project (B, H, W, E) => (B, H, W, E)
        return self._project_block(__outputs)

    def get_config(self) -> dict:
        __config = super(VitDiffusionModel, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
