import tensorflow as tf
import numpy      as np

from qfast.native.uq import create_wolfram_uq_script, convert_utry_to_wolfram_str


class TestUQCreateWolframUQScript ( tf.test.TestCase ):

    def test_uq_create_wolfram_uq_script ( self ):
        utry1 = np.array( [ [ 1, 0 ],
                            [ 0, 1 ] ], dtype = np.complex128 )

        wolfram_str = convert_utry_to_wolfram_str( utry1 )
        lines = create_wolfram_uq_script( utry1 ).splitlines()
        self.assertTrue( "QI" in lines[0] )
        self.assertTrue( "UniversalQCompiler" in lines[1] )
        self.assertTrue( "utry = " + wolfram_str in lines[2] )
        self.assertTrue( "gatelist = QSD[utry]" in lines[3] )
        self.assertTrue( "Print[gatelist]" in lines[4] )


if __name__ == '__main__':
    tf.test.main()
