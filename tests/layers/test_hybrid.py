import tensorflow as tf

import tr1cot.layers.hybrid

# PATCHING #####################################################################

class HybridBlockTest(tf.test.TestCase):
    def setUp(self):
        super(HybridBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 2, 'block_num': 2, 'head_num': 2, 'dropout_val': 0.1, 'epsilon_val': 1e-6,},},
            {
                'inputs': tf.ones((2, 4, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 2, 'block_num': 2},},
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'patch_dim': 1, 'block_num': 4},},
            {
                'inputs': tf.cast([16 * [16 * [8 * [__i + 1]]] for __i in range(2)], dtype=tf.float32),
                'args': {},},]

    def test_shapes_and_dtypes(self):
        for __case in self._null_cases:
            __layer = tr1cot.layers.hybrid.HybridBlock(**__case['args'])
            # end-to-end
            __outputs = __layer(__case['inputs'], training=False)
            self.assertEqual(tuple(__case['inputs'].shape), tuple(__outputs.shape))
            self.assertEqual(tf.float32, __outputs.dtype)
            # test each block
            __outputs = __case['inputs']
            for __block in __layer._blocks:
                __outputs = __block(__outputs)
                self.assertEqual(tuple(__case['inputs'].shape), tuple(__outputs.shape))
