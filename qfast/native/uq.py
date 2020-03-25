"""
This module implements UniversalQ as a native tool plugin to QFAST.
"""

import os
from os import path
import subprocess

import numpy as np


def convert_utry_to_wolfram_str ( utry ):
    """
    Returns the Wolfram representation of utry as a string.

    Args:
        utry (np.ndarray): The unitary matrix to convert.

    Returns:
        str_out (str): Wolfram string representation of utry
    """

    str_out = np.array2string( utry, precision = 16, separator = ",",
                               max_line_width = 100 * len( utry ),
                               threshold = 100 * len( utry ) * len( utry ),
                               floatmode = "fixed",
                               formatter = { 'all' : lambda x: str(x) } )
    str_out = str_out.replace( "\n", "" ).replace( " ", "" )
    str_out = str_out.replace( "[", "{" ).replace( "]", "}" )
    str_out = str_out.replace( "j", "*I" )
    return str_out


def create_wolfram_uq_script ( utry ):
    """
    Returns a Wolfram script that runs UQ's QSD on the utry.

    Args:
        utry (np.ndarray): The script is written for this unitary matrix

    Returns:
        (str): Wolfram string representation of the script
    """

    utry_wolfram_str = convert_utry_to_wolfram_str( utry )

    uq_dir = path.abspath( path.dirname( __file__ ) )
    QI_file = path.join( uq_dir, "QI.m" )
    UQ_file = path.join( uq_dir, "UniversalQCompiler.m" )

    if ( not path.isdir( uq_dir )
         or not path.isfile( QI_file )
         or not path.isfile( UQ_file ) ):
        raise RuntimeError( "Unable to find UniversalQ Installation." )


    return ( "Needs[\"QI`\", \"" + QI_file + "\" ]\n"
             "Needs[\"UniversalQCompiler`\", \"" + UQ_file + "\" ]\n"
             "utry = " + utry_wolfram_str + "\n"
             "gatelist = QSD[utry]\n"
             "Print[gatelist]\n" )


def run_uq_compiler ( utry ):
    """
    Returns the output from running the uq compiler on the utry.

    Args:
        utry (np.ndarray): The unitary to compile

    Returns:
        uq_out (str): Output circuit in UniversalQ's circuit language
    """

    script = create_wolfram_uq_script( utry )

    script_file = "uq_script.m"
    with open( script_file, 'w' ) as f:
        f.write( script )

    uq_out = subprocess.check_output( [ "math", "-script", script_file ],
                                      universal_newlines = True )

    os.remove( script_file )

    return uq_out


def parse_uq_out ( uq_out ):
    """
    Parses UniversalQ's output language and converts to qasm.

    Args:
        uq_out (np.ndarray): The output to parse

    Returns:
        (str): Output qasm
    """

    # Parse the output into gates
    gates = []
    for line in uq_out.splitlines():
        line = line.replace( ",\n", "" ).replace( ", \n", "" )
        line = line.replace( "\n", "" )
        line = line.replace( "{{", "{" ).replace( "}}", "}" )
        line = line.replace( "{", "" )
        line = line.replace( "},", ";" ).replace( "}", ";" )
        line = line.replace( " ", "" )
        for gate in line.split(";"):
            parsed_gate = gate.split(",")

            if len( parsed_gate ) != 3:
                continue

            parsed_gate = [ int( parsed_gate[0] ),
                            parsed_gate[1],
                            int( parsed_gate[2] ) - 1 ]

            if parsed_gate[0] == 0:
                parsed_gate[1] = int( parsed_gate[1] ) - 1
            else:
                parsed_gate[1] = parsed_gate[1].replace( "Pi", "3.14159265358979324" )

                if "/" in parsed_gate[1]:
                    num, dem = parsed_gate[1].split( "/" )

                    if "*" in num:
                        left, right = num.split( "*" )
                        left = left.replace( "(", "" )
                        right = right.replace( ")", "" )
                        num = float( left ) * float( right )

                    parsed_gate[1] = float( num ) / float( dem )

                elif "*^" in parsed_gate[1]:
                    exp = int( parsed_gate[1].split("*^")[1] )
                    parsed_gate[1] = float( parsed_gate[1].split("*^")[0] ) * (10 ** exp)

                parsed_gate[1] = float( parsed_gate[1] )

            gates.append( parsed_gate )

    # Calculate Circuit Size
    max_qubit = 0
    for gate in gates:
        if gate[2] >= max_qubit:
            max_qubit = gate[2]
        if gate[0] == 0 and gate[1] >= max_qubit:
            max_qubit = gate[1]

    # Build QASM
    qasm_builder = []
    qasm_builder.append( "OPENQASM 2.0;\n"
                         "include \"qelib1.inc\";\n"
                         "qreg q[" + str( max_qubit + 1 ) + "];\n" )

    for gate in gates:
        if gate[0] == 0:
            qasm_builder.append( "cx q[" + str( gate[1] ) + "],q["
                                 + str( gate[2] ) + "];\n" )
        elif gate[0] == 1:
            qasm_builder.append( "rx(" + str( -gate[1] ) + ") q[" + str( gate[2] ) + "];\n" )
        elif gate[0] == 2:
            qasm_builder.append( "ry(" + str( -gate[1] ) + ") q[" + str( gate[2] ) + "];\n" )
        elif gate[0] == 3:
            qasm_builder.append( "rz(" + str( -gate[1] ) + ") q[" + str( gate[2] ) + "];\n" )

    return ''.join( qasm_builder )


def get_native_block_size():
    """
    The maximum size of a unitary matrix (in qubits) that can be
    decomposed with this module.

    Returns:
        (int): The qubit count this module can handle.
    """

    # 3 is the optimal amount to start calling uq
    return 3


def synthesize ( utry ):
    """
    Synthesis function with UniversalQ.

    Args:
        utry (np.ndarray): The unitary matrix to synthesize.

    Returns
        qasm (str): The synthesized QASM output.
    """

    if not isinstance( utry, np.ndarray ):
        raise TypeError( "utry must be a np.ndarray." )

    if len( utry.shape ) != 2:
        raise TypeError( "utry must be a matrix." )

    if utry.shape[0] != 2 ** get_native_block_size():
        raise ValueError( "utry has incorrect dimensions." )

    if utry.shape[1] != 2 ** get_native_block_size():
        raise ValueError( "utry has incorrect dimensions." )

    if ( not np.allclose( utry.conj().T @ utry, np.identity( len( utry ) ),
                          rtol = 0, atol = 1e-14 )
         or
         not np.allclose( utry @ utry.conj().T, np.identity( len( utry ) ),
                          rtol = 0, atol = 1e-14 ) ):
        raise ValueError( "utry must be a unitary matrix." )

    return parse_uq_out( run_uq_compiler( utry ) )
