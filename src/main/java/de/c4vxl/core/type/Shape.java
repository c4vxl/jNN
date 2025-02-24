package de.c4vxl.core.type;

import de.c4vxl.core.utils.TensorUtils;

import java.util.Arrays;

public class Shape {
    public int[] dimensions;

    public Shape(int... shape) {
        this.dimensions = shape;
    }

    /**
     * Returns the amount of dimensions this Shape has
     */
    public int rank() { return dimensions.length; }

    /**
     * Returns the amount of data this Tensor can store
     */
    public int size() { return TensorUtils.shapeToSize(this.dimensions); }


    @Override
    public String toString() {
        return Arrays.toString(dimensions);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Shape && Arrays.equals(((Shape) o).dimensions, this.dimensions);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dimensions);
    }
}
