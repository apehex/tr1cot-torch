import functools
import math

import tensorflow as tf

# CONSTANTS ####################################################################

# meta
EPSILON = 1e-5
DROPOUT = 0.0
MOMENTUM = 0.99

# convolution blocks
EXPAND_RATE = 4
SQUEEZE_RATE = 4
SAMPLING_DIM = 1
POOLING_DIM = 5
KERNEL_DIM = 3

# RESIDUAL #####################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, latent_dim: int=None, **kwargs) -> None:
        super(ResidualBlock, self).__init__(**kwargs)
        # save the config to init later
        self._config = {'latent_dim': latent_dim,}
        # create when the input shape is known
        self._norm = None
        self._conv0 = None
        self._conv1 = None
        self._conv2 = None

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)[:-1] + (self._config['latent_dim'],)

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        __filters = self._config['latent_dim'] or __shape[-1]
        __groups = 2 ** int(0.5 * math.log2(__shape[-1]))
        __args = {'filters': __filters, 'strides': 1, 'use_bias': True, 'padding': 'same', 'data_format': 'channels_last',}
        # update
        self._config['latent_dim'] = __filters
        # init
        self._norm = tf.keras.layers.GroupNormalization(groups=__groups, axis=-1, center=True, scale=True)
        self._conv0 = tf.keras.layers.Conv2D(kernel_size=1, **__args)
        self._conv1 = tf.keras.layers.Conv2D(kernel_size=3, activation='silu', **__args)
        self._conv2 = tf.keras.layers.Conv2D(kernel_size=3, **__args)
        # build
        self._norm.build(__shape)
        self._conv0.build(__shape)
        self._conv1.build(__shape)
        self._conv2.build(self._conv1.compute_output_shape(__shape))
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return self._conv0(inputs) + self._conv2(self._conv1(self._norm(inputs, training=training)))

    def get_config(self) -> dict:
        __config = super(ResidualBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# SE ###########################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class SqueezeAndExcitationBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        pooling_dim: int=POOLING_DIM,
        activation='silu',
        **kwargs
    ) -> None:
        # init
        super(SqueezeAndExcitationBlock, self).__init__(**kwargs)
        # normalize
        __pooling_dim = (pooling_dim, pooling_dim) if isinstance(pooling_dim, int) else tuple(pooling_dim)[:2]
        # config
        self._config = {
            'hidden_dim': hidden_dim,
            'pooling_dim': __pooling_dim,
            'activation': activation,}

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)

    def build(self, input_shape: tuple) -> None:
        # common args
        __args = {
            'kernel_size': 1,
            'strides': 1,
            'use_bias': True,
            'padding': 'same',
            'data_format': 'channels_last',
            'kernel_initializer': 'glorot_normal',
            'bias_initializer': 'zeros'}
        # layers
        self._pool = tf.keras.layers.AveragePooling2D(pool_size=self._config['pooling_dim'], strides=1, padding='same', data_format='channels_last')
        self._reduce = tf.keras.layers.Conv2D(self._config['hidden_dim'], activation=self._config['activation'], **__args)
        self._expand = tf.keras.layers.Conv2D(input_shape[-1], activation=None, **__args)
        # build
        self._pool.build(input_shape)
        self._reduce.build(input_shape)
        self._expand.build(list(input_shape)[:-1] + [self._config['hidden_dim']])
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # compute the switch
        __gate = self._expand(self._reduce(self._pool(inputs)))
        # dynamic selection of the features
        return tf.sigmoid(__gate) * inputs

    def get_config(self) -> dict:
        __config = super(SqueezeAndExcitationBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# MOBILE ######################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class MobileNetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        feature_dim: int,
        expand_rate: int=EXPAND_RATE,
        squeeze_rate: int=SQUEEZE_RATE,
        dropout_rate: float=DROPOUT,
        sampling_dim: int=SAMPLING_DIM, # 1 = no downsampling
        pooling_dim: int=POOLING_DIM,
        kernel_dim: int=KERNEL_DIM,
        **kwargs
    ) -> None:
        # init
        super(MobileNetBlock, self).__init__(**kwargs)
        # normalize
        __sampling_dim = (sampling_dim, sampling_dim) if isinstance(sampling_dim, int) else tuple(sampling_dim)[:2]
        __pooling_dim = (pooling_dim, pooling_dim) if isinstance(pooling_dim, int) else tuple(pooling_dim)[:2]
        __kernel_dim = (kernel_dim, kernel_dim) if isinstance(kernel_dim, int) else tuple(kernel_dim)[:2]
        # save config for exports
        self._config = {
            'feature_dim': feature_dim,
            'expand_rate': expand_rate,
            'squeeze_rate': squeeze_rate,
            'dropout_rate': dropout_rate,
            'sampling_dim': __sampling_dim,
            'pooling_dim': __pooling_dim,
            'kernel_dim': __kernel_dim,}

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        __batch_dim, __height_dim, __width_dim = tuple(input_shape)[:3]
        return (__batch_dim, __height_dim // self._config['sampling_dim'][0], __width_dim // self._config['sampling_dim'][-1], self._config['feature_dim'])

    def build(self, input_shape: tuple) -> None:
        __batch_dim, __height_dim, __width_dim = tuple(input_shape)[:3]
        # successive feature dimensions
        __feature_dim = self._config['feature_dim']
        __expand_dim = __feature_dim * self._config['expand_rate']
        __squeeze_dim = max(1, __feature_dim // self._config['squeeze_rate'])
        # space and time dimensions
        __sampling_dim = self._config['sampling_dim']
        __pooling_dim = self._config['pooling_dim']
        __kernel_dim = self._config['kernel_dim']
        # common args for all convs
        __conv_args = {
            'kernel_size': 1,
            'strides': 1,
            'padding': 'same',
            'data_format': 'channels_last',
            'kernel_initializer': 'glorot_normal',
            'bias_initializer': 'zeros',}
        # downsampling args
        __samp_args = {
            'kernel_size': __kernel_dim,
            'strides': __sampling_dim,
            'padding': 'same',
            'data_format': 'channels_last',
            'kernel_initializer': 'glorot_normal',
            'bias_initializer': 'zeros',}
        # common args for all norms
        __norm_args = {
            'axis': -1,
            'momentum': MOMENTUM,
            'epsilon': EPSILON,
            'center': True,
            'scale': True,}
        # batch norms
        self._norm_pre = tf.keras.layers.BatchNormalization(**__norm_args)
        self._norm_expand = tf.keras.layers.BatchNormalization(**__norm_args)
        self._norm_sample = tf.keras.layers.BatchNormalization(**__norm_args)
        # activation
        self._func_activation = tf.keras.activations.gelu
        # shortcut path: downsample + project
        self._conv_shortcut = tf.keras.layers.Conv2D(filters=__feature_dim, use_bias=True, **__samp_args)
        # residual path: expand + downsample + select + shrink
        __samp_args['depthwise_initializer'] = __samp_args.pop('kernel_initializer')
        self._conv_expand = tf.keras.layers.Conv2D(filters=__expand_dim, use_bias=False, **__conv_args)
        self._conv_sample = tf.keras.layers.DepthwiseConv2D(use_bias=False, **__samp_args)
        self._dropout = tf.keras.layers.Dropout(self._config['dropout_rate'])
        self._conv_select = SqueezeAndExcitationBlock(hidden_dim=__squeeze_dim, pooling_dim=__pooling_dim)
        self._conv_shrink = tf.keras.layers.Conv2D(filters=__feature_dim, use_bias=True, **__conv_args)
        # build
        self._conv_shortcut.build(input_shape)
        self._norm_pre.build(input_shape)
        self._conv_expand.build(input_shape)
        self._norm_expand.build((__batch_dim, __height_dim, __width_dim, __expand_dim))
        self._conv_sample.build((__batch_dim, __height_dim, __width_dim, __expand_dim))
        self._norm_sample.build((__batch_dim, __height_dim // __sampling_dim[0], __width_dim // __sampling_dim[-1], __expand_dim))
        self._dropout.build((__batch_dim, __height_dim // __sampling_dim[0], __width_dim // __sampling_dim[-1], __expand_dim))
        self._conv_select.build((__batch_dim, __height_dim // __sampling_dim[0], __width_dim // __sampling_dim[-1], __expand_dim))
        self._conv_shrink.build((__batch_dim, __height_dim // __sampling_dim[0], __width_dim // __sampling_dim[-1], __expand_dim))
        # register
        self.built = True

    def _shortcut_branch(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return self._conv_shortcut(inputs, training=training)

    def _residual_branch(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        __outputs = self._norm_pre(inputs, training=training)
        # expand the features
        __outputs = self._conv_expand(__outputs, training=training)
        __outputs = self._norm_expand(__outputs, training=training)
        __outputs = self._func_activation(__outputs)
        # learn spatial patterns + downsample
        __outputs = self._conv_sample(__outputs, training=training)
        __outputs = self._norm_sample(__outputs, training=training)
        __outputs = self._func_activation(__outputs)
        # drop random neurons
        __outputs = self._dropout(__outputs, training=training)
        # dynamic selection of the relevant features
        __outputs = self._conv_select(__outputs, training=training)
        # shrink back to the common dim
        return self._conv_shrink(__outputs, training=training)

    def call(self, inputs: tf.Tensor, training: bool=False, survival_prob=None) -> tf.Tensor:
        return self._shortcut_branch(inputs, training=training) + self._residual_branch(inputs, training=training)

    def get_config(self) -> dict:
        __config = super(MobileNetBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# DOWNSAMPLING #################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class DownBlock(tf.keras.layers.Layer):
    def __init__(self, block_num: int, block_dim: int, **kwargs) -> None:
        super(DownBlock, self).__init__(**kwargs)
        # save the config to init later
        self._config = {'block_num': block_num, 'block_dim': block_dim,}
        # layers
        self._blocks = []

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __x, __b: __b.compute_output_shape(__x), self._blocks, input_shape)

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._blocks = [ResidualBlock(latent_dim=self._config['block_dim']) for _ in range(self._config['block_num'])]
        self._blocks.append(tf.keras.layers.AveragePooling2D(pool_size=2))
        # build
        for __b in self._blocks:
            __b.build(__shape)
            __shape = __b.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __b: __b(__x, training=training), self._blocks, inputs)

    def get_config(self) -> dict:
        __config = super(DownBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# UPSAMPLING ###################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class UpBlock(tf.keras.layers.Layer):
    def __init__(self, block_num: int, block_dim: int, **kwargs) -> None:
        super(UpBlock, self).__init__(**kwargs)
        # save the config to init later
        self._config = {'block_num': block_num, 'block_dim': block_dim,}
        # layers
        self._blocks = []

    def compute_output_shape(self, inputs_shape: tuple) -> tuple:
        return functools.reduce(lambda __x, __b: __b.compute_output_shape(__x), self._blocks, inputs_shape)

    def build(self, inputs_shape: tuple) -> None:
        __shape = tuple(inputs_shape)
        # init
        self._blocks = [tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')]
        self._blocks.extend([ResidualBlock(latent_dim=self._config['block_dim']) for _ in range(self._config['block_num'])])
        # build
        for __b in self._blocks:
            __b.build(__shape)
            __shape = __b.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> list:
        return functools.reduce(lambda __x, __b: __b(__x, training=training), self._blocks, inputs)

    def get_config(self) -> dict:
        __config = super(UpBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
