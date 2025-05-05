import tensorflow as tf

import tr1cot.layers.transformer

# PATCHING #####################################################################

class PatchTransformerBlockTest(tf.test.TestCase):
    def setUp(self):
        super(PatchTransformerBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 2, 'head_num': 2, 'dropout_rate': 0.1, 'epsilon_rate': 1e-6,},
                'shapes': {'patched': (2, 8, 8, 32), 'merged': (2, 64, 32), 'attended': (2, 64, 32), 'divided': (2, 8, 8, 32), 'unpatched': (2, 16, 16, 8)},
                'outputs': tf.ones((2, 16, 16, 8), dtype=tf.float32),},
            {
                'inputs': tf.ones((2, 4, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 2, 'head_num': 2},
                'shapes': {'patched': (2, 2, 8, 32), 'merged': (2, 16, 32), 'attended': (2, 16, 32), 'divided': (2, 2, 8, 32), 'unpatched': (2, 4, 16, 8)},
                'outputs': tf.ones((2, 4, 16, 8), dtype=tf.float32),},
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 1, 'head_num': 4},
                'shapes': {'patched': (2, 16, 16, 8), 'merged': (2, 256, 8), 'attended': (2, 256, 8), 'divided': (2, 16, 16, 8), 'unpatched': (2, 16, 16, 8)},
                'outputs': tf.ones((2, 16, 16, 8), dtype=tf.float32),},
            {
                'inputs': tf.cast([16 * [16 * [8 * [__i + 1]]] for __i in range(2)], dtype=tf.float32),
                'args': {},
                'shapes': {'patched': (2, 4, 4, 128), 'merged': (2, 16, 128), 'attended': (2, 16, 128), 'divided': (2, 4, 4, 128), 'unpatched': (2, 16, 16, 8)},
                'outputs': tf.cast([16 * [16 * [8 * [__i + 1]]] for __i in range(2)], dtype=tf.float32),},]

    def test_shapes_and_dtypes(self):
        for __case in self._null_cases:
            __layer = tr1cot.layers.transformer.PatchTransformerBlock(**__case['args'])
            # end-to-end
            __outputs = __layer(__case['inputs'], training=False)
            self.assertEqual(tuple(__case['shapes']['unpatched']), tuple(__outputs.shape))
            self.assertEqual(__case['outputs'].dtype, __outputs.dtype)
            # patching space axes
            __outputs = __layer._patch_space(__case['inputs'])
            self.assertEqual(tuple(__case['shapes']['patched']), tuple(__outputs.shape))
            # merging space axes
            __outputs = __layer._merge_space(__outputs)
            self.assertEqual(tuple(__case['shapes']['merged']), tuple(__outputs.shape))
            # attending merged axis
            __outputs = __layer._attend_space(query=__outputs, key=__outputs, value=__outputs, training=False, use_causal_mask=False)
            self.assertEqual(tuple(__case['shapes']['attended']), tuple(__outputs.shape))
            # dividing the space axis
            __outputs = __layer._split_space(__outputs)
            self.assertEqual(tuple(__case['shapes']['divided']), tuple(__outputs.shape))
            # unpatching
            __outputs = __layer._unpatch_space(__outputs)
            self.assertEqual(tuple(__case['shapes']['unpatched']), tuple(__outputs.shape))

    def test_outputs_equal_inputs_when_constant(self): # outputs = inputs + layer(inputs), the bias is initialized to zero and the norm cancels the product
        for __case in self._null_cases:
            __layer = tr1cot.layers.transformer.PatchTransformerBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            self.assertAllEqual(__outputs, __case['outputs'])

# DOWN SAMPLING ################################################################

class DownBlockTest(tf.test.TestCase):
    def setUp(self):
        super(DownBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 2,},
                'shapes': {'patched': (2, 8, 8, 32), 'compressed': (2, 8, 8, 16)},},
            {
                'inputs': tf.ones((2, 4, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 2,},
                'shapes': {'patched': (2, 2, 8, 32), 'compressed': (2, 2, 8, 16)},},
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 1,},
                'shapes': {'patched': (2, 16, 16, 8), 'compressed': (2, 16, 16, 8)},},]

    def test_shapes_and_dtypes(self):
        for __case in self._null_cases:
            __layer = tr1cot.layers.transformer.DownBlock(**__case['args'])
            # build
            __layer(__case['inputs'], training=False)
            # patching the space axes
            __outputs = __layer._patch(__case['inputs'])
            self.assertEqual(tuple(__case['shapes']['patched']), tuple(__outputs.shape))
            # compressing the features
            __outputs = __layer._dense(__outputs)
            self.assertEqual(tuple(__case['shapes']['compressed']), tuple(__outputs.shape))

# UP SAMPLING ##################################################################

class UpBlockTest(tf.test.TestCase):
    def setUp(self):
        super(UpBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 2,},
                'shapes': {'expanded': (2, 16, 16, 16), 'unpatched': (2, 32, 32, 4)},},
            {
                'inputs': tf.ones((2, 4, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 4,},
                'shapes': {'expanded': (2, 4, 16, 32), 'unpatched': (2, 16, 64, 2)},},
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 1,},
                'shapes': {'expanded': (2, 16, 16, 8), 'unpatched': (2, 16, 16, 8)},},]

    def test_shapes_and_dtypes(self):
        for __case in self._null_cases:
            __layer = tr1cot.layers.transformer.UpBlock(**__case['args'])
            # build
            __layer(__case['inputs'], training=False)
            # expanding the features
            __outputs = __layer._dense(__case['inputs'])
            self.assertEqual(tuple(__case['shapes']['expanded']), tuple(__outputs.shape))
            # unpatching the spatial axes
            __outputs = __layer._patch(__outputs)
            self.assertEqual(tuple(__case['shapes']['unpatched']), tuple(__outputs.shape))
