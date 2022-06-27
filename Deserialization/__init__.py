from type_declarations import *
from math_primitives.Vector import Vector
from data_types.PrimaryData import PrimaryDataNumeric
import Vector as VectorMethods
import Float as FloatMethods
import PrimaryDataNumeric as PrimaryDataNumericMethods

# Add deserialization extension methods here

Vector.deserialize = VectorMethods.deserialize
float.deserialize = FloatMethods.deserialize

PrimaryDataNumeric.deserialize = PrimaryDataNumericMethods.deserialize
