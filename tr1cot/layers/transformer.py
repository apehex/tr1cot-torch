import math

import tensorflow as tf

import mlable.blocks.transformer
import mlable.layers.embedding
import mlable.layers.shaping
import mlable.blocks.shaping

# CONSTANTS ####################################################################

# meta
EPSILON = 1e-5
DROPOUT = 0.0

# attention blocks
PATCH_DIM = 0 # P
HEAD_NUM = 0 # N

# TIME ATTENTION ###############################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class PatchTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_dim: int=PATCH_DIM,
        head_num: int=HEAD_NUM,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(PatchTransformerBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'patch_dim': patch_dim,
            'head_num': head_num,
            'dropout_rate': dropout_rate,
            'epsilon_rate': epsilon_rate,}
        # layers
        self._patch_space = None
        self._unpatch_space = None
        self._merge_space = None
        self._split_space = None
        self._attend_space = None

    def compute_root_dim(self, input_dim: int) -> int:
        return 2 ** int(0.5 * math.log2(input_dim))

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)

    def build(self, input_shape: tuple) -> None:
        __batch_dim, __height_dim, __width_dim, __feature_dim = input_shape
        # internal dimensions
        __head_num = self._config['head_num'] or self.compute_root_dim(__feature_dim)
        __patch_dim = self._config['patch_dim'] or min(self.compute_root_dim(__height_dim), self.compute_root_dim(__width_dim))
        __space_dim = (__height_dim * __width_dim) // (__patch_dim ** 2)
        __latent_dim = __patch_dim * __patch_dim * __feature_dim
        __hidden_dim = 4 * __latent_dim
        __head_dim = __latent_dim // __head_num
        # init the layers
        self._patch_space = mlable.blocks.shaping.PixelPacking(patch_dim=__patch_dim, height_axis=1, width_axis=2)
        self._unpatch_space = mlable.blocks.shaping.PixelShuffle(patch_dim=__patch_dim, height_axis=1, width_axis=2)
        self._merge_space = mlable.layers.shaping.Merge(axis=1, right=True)
        self._split_space = mlable.layers.shaping.Divide(axis=1, factor=__width_dim // __patch_dim, insert=True, right=True)
        self._attend_space = mlable.blocks.transformer.ResidualDecoderBlock(head_num=__head_num, key_dim=__head_dim, value_dim=__head_dim, hidden_dim=__hidden_dim, attention_axes=[1], epsilon=self._config['epsilon_rate'], dropout_rate=self._config['dropout_rate'], use_position=False, use_bias=True, center=True, scale=True)
        # built with the specific shape, at each step
        self._patch_space.build(input_shape)
        self._merge_space.build((__batch_dim, __height_dim // __patch_dim, __width_dim // __patch_dim, __latent_dim))
        self._attend_space.build(query_shape=(__batch_dim, __space_dim, __latent_dim), key_shape=(__batch_dim, __space_dim, __latent_dim), value_shape=(__batch_dim, __space_dim, __latent_dim))
        self._split_space.build((__batch_dim, __space_dim, __latent_dim))
        self._unpatch_space.build((__batch_dim, __height_dim // __patch_dim, __width_dim // __patch_dim, __latent_dim))
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # global patching (B, H, W, E) => (B, H/P, W/P, PPE)
        __outputs = self._patch_space(inputs)
        # merge the patch and feature axes (B, H/P, W/P, PPE) => (B, HW/PP, PPE)
        __outputs = self._merge_space(__outputs)
        # space attention (across grids) (B, HW/PP, PPE)
        __outputs = self._attend_space(query=__outputs, key=__outputs, value=__outputs, training=training, use_causal_mask=False)
        # split patch features (B, HW/PP, PPE) => (B, H/P, W/P, PPE)
        __outputs = self._split_space(__outputs)
        # swap the axes back (B, H/P, W/P, PPE) => (B, H, W, E)
        return self._unpatch_space(__outputs)

    def get_config(self) -> dict:
        __config = super(PatchTransformerBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# DOWNSAMPLING #################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class DownBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_dim: int,
        **kwargs
    ) -> None:
        # init
        super(DownBlock, self).__init__(**kwargs)
        # config
        self._config = {'patch_dim': patch_dim,}
        # layers
        self._patch = None
        self._dense = None

    def build(self, input_shape: tuple) -> None:
        __batch_dim, __height_dim, __width_dim, __feature_dim = tuple(input_shape)
        # downsample with patching
        self._patch = mlable.blocks.shaping.PixelPacking(
            patch_dim=self._config['patch_dim'],
            height_axis=1,
            width_axis=2)
        # compress the features
        self._dense = tf.keras.layers.Dense(
            units=__feature_dim * self._config['patch_dim'], # F*P instead of F*P*P
            activation=None,
            use_bias=True)
        # build
        self._patch.build(input_shape)
        self._dense.build(self._patch.compute_output_shape(input_shape))
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return self._dense.compute_output_shape(self._patch.compute_output_shape(input_shape))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # downsample (B, H, W, E) => (B, H/P, W/P, PPE)
        __outputs = self._patch(inputs)
        # compress (B, H/P, W/P, PPE) => (B, H/P, W/P, PE)
        return self._dense(__outputs)

    def get_config(self) -> dict:
        __config = super(DownBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# UP SAMPLE ####################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class UpBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_dim: int,
        **kwargs
    ) -> None:
        # init
        super(UpBlock, self).__init__(**kwargs)
        # config
        self._config = {'patch_dim': patch_dim,}
        # layers
        self._dense = None
        self._patch = None

    def build(self, input_shape: tuple) -> None:
        __batch_dim, __height_dim, __width_dim, __feature_dim = tuple(input_shape)
        # expand the features
        self._dense = tf.keras.layers.Dense(
            units=__feature_dim * self._config['patch_dim'],
            activation=None,
            use_bias=True)
        # upample with unpatching
        self._patch = mlable.blocks.shaping.PixelShuffle(
            patch_dim=self._config['patch_dim'],
            height_axis=1,
            width_axis=2)
        # build
        self._dense.build(input_shape)
        self._patch.build(self._dense.compute_output_shape(input_shape))
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return self._patch.compute_output_shape(self._dense.compute_output_shape(input_shape))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # expand (B, H/P, W/P, PE) => (B, H/P, W/P, PPE)
        __outputs = self._dense(inputs)
        # upsample (B, H/P, W/P, PPE) => (B, H, W, E)
        return self._patch(__outputs)

    def get_config(self) -> dict:
        __config = super(UpBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
