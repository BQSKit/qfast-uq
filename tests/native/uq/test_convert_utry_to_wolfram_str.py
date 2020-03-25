import tensorflow as tf
import numpy      as np

from qfast.native.uq import convert_utry_to_wolfram_str


class TestUQConvertUtryToWolframStr ( tf.test.TestCase ):

    def test_uq_convert_utry_to_wolfram_str_1 ( self ):
        utry1 = np.array( [ [ 1, 0 ],
                            [ 0, 1 ] ], dtype = np.complex128 )

        wolfram_str = convert_utry_to_wolfram_str( utry1 )
        self.assertEqual( wolfram_str, "{{(1+0*I),0*I},{0*I,(1+0*I)}}" )

    def test_uq_convert_utry_to_wolfram_str_2 ( self ):
        utry2 = np.array( [ [ 1, 0, 0, 0 ],
                            [ 0, 1, 0, 0 ],
                            [ 0, 0, 0, 1 ],
                            [ 0, 0, 1, 0 ] ], dtype = np.complex128 )

        wolfram_str = convert_utry_to_wolfram_str( utry2 )
        self.assertEqual( wolfram_str, "{{(1+0*I),0*I,0*I,0*I},{0*I,(1+0*I),0*I,0*I},{0*I,0*I,0*I,(1+0*I)},{0*I,0*I,(1+0*I),0*I}}" )


if __name__ == '__main__':
    tf.test.main()
