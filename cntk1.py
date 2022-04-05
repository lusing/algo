# Import CNTK library
import cntk
import numpy as np

# Version
print('CNTK version:', cntk.__version__)

# Check GPU
from cntk.device import try_set_default_device, gpu
if try_set_default_device(gpu(0)):
    print('GPU device is enabled!')

# Print devices
print('All available devices:', cntk.all_devices())

# Simple math operation
print('Welcome to CNTK world!')

#################################
#### Mathematical operations ####
#################################

# Initial definition
a = [1, 2, 3]
b = [3, 2, 1]

# Get the type of the variable
print(type(a))

# Subtraction
print(cntk.minus(a,b).eval())

# Additive
print(cntk.plus(a,b).eval())

# Element-wise division
print(cntk.element_divide(a,b).eval())


# Defining variable
variable = cntk.input_variable((2), np.float32)
print(variable)
