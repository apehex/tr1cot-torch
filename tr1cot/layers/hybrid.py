import functools

import tensorflow as tf

import tr1cot.layers.convolution
import tr1cot.layers.transformer

# CONSTANTS ####################################################################

# meta constants
DROPOUT = 0.0
EPSILON = 1e-5
MOMENTUM = 0.99

# encoder and decoder
PATCH_DIM = 4 # P
BLOCK_NUM = 4 # N
HEAD_NUM = 4 # H

# TRANSFORMER ##################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class HybridTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_dim: int=PATCH_DIM,
        block_num: int=BLOCK_NUM,
        head_num: int=HEAD_NUM,
        dropout_val: float=DROPOUT,
        epsilon_val: float=EPSILON,
        **kwargs
    ) -> None:
        super(HybridTransformerBlock, self).__init__(**kwargs)
        # save the config to init later
        self._config = {
            'block_num': block_num,
            'head_num': head_num,
            'patch_dim': patch_dim,
            'dropout_val': dropout_val,
            'epsilon_val': epsilon_val,}
        # layers
        self._blocks = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._blocks = [
            tr1cot.layers.convolution.ResidualBlock(latent_dim=input_shape[-1])
            for _ in range(self._config['block_num'])] + [
            tr1cot.layers.transformer.PatchTransformerBlock(
                patch_dim=self._config['patch_dim'],
                head_num=self._config['head_num'],
                dropout_rate=self._config['dropout_val'],
                epsilon_rate=self._config['epsilon_val'])]
        # build
        for __b in self._blocks:
            __b.build(__shape)
            __shape = __b.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __x, __b: __b.compute_output_shape(__x), self._blocks, input_shape)

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __b: __b(__x, training=training), self._blocks, inputs)

    def get_config(self) -> dict:
        __config = super(HybridTransformerBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)