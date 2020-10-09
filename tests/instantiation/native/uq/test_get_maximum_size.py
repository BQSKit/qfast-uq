import numpy    as np
import unittest as ut

from qfast.instantiation.native.uq import UQTool


class TestUQGetMaximumSize ( ut.TestCase ):

    def test_uq_get_maximum_size ( self ):
        uqtool = UQTool()
        block_size = uqtool.get_maximum_size()
        self.assertEqual( block_size, 3 )


if __name__ == '__main__':
    ut.main()
