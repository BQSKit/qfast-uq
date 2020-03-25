import tensorflow as tf
import numpy      as np

from qfast.native.uq import get_native_block_size


class TestUQGetNativeBlockSize ( tf.test.TestCase ):

    def test_uq_get_native_block_size ( self ):
        block_size = get_native_block_size()
        self.assertEqual( block_size, 3 )


if __name__ == '__main__':
    tf.test.main()
