#########################################################################
# File Name: __init__.py
# Description: Manages the initialization of qubit modules.
#              Includes functionality for dynamically importing qubit-related classes 
#              and maintaining module information.
#########################################################################

from library.qubits.circlemon import Circlemon
from library.qubits.custom_qubit import CustomQubit
from library.qubits.transmon_interdigitated import TransmonInterdigitated
from library.qubits.transmon_rotate import TransmonRotate
from library.qubits.transmon_teeth import TransmonTeeth
from library.qubits.transmon import Transmon
from library.qubits.xmon import Xmon
from library.qubits.xmon_rotate import XmonRotate
from library.qubits.transmon_benzheng import TransmonBenzheng
from library.qubits.TransmonConcentric_1 import Transmonconcentric1
from library.qubits.TransmonConcentricType2_1 import Transmonconcentrictype21
from library.qubits.TransmonCross_1 import Transmoncross1
from library.qubits.TransmonCrossFL_1 import Transmoncrossfl1
from library.qubits.TransmonInterdigitated_1 import Transmoninterdigitated1
from library.qubits.TransmonPocket6_1 import Transmonpocket61
from library.qubits.TransmonPocket_1 import Transmonpocket1
from library.qubits.TransmonPocketCL_1 import Transmonpocketcl1
from library.qubits.TransmonPocketTeeth_1 import Transmonpocketteeth1
from library.qubits.SQUID_LOOP_1 import SquidLoop1
from library.qubits.StarQubit_1 import Starqubit1

module_name_list = [
    "circlemon", 
    "custom_qubit",  
    "transmon_interdigitated", 
    "transmon_rotate", 
    "transmon_teeth", 
    "transmon", 
    "xmon",
    "xmon_rotate",
    "transmon_benzheng",
    "TransmonConcentric_1",
    "TransmonConcentricType2_1",
    "TransmonCross_1",
    "TransmonCrossFL_1",
    "TransmonInterdigitated_1",
    "TransmonPocket6_1",
    "TransmonPocket_1",
    "TransmonPocketCL_1",
    "TransmonPocketTeeth_1",
    "SQUID_LOOP_1",
    "StarQubit_1"

]
