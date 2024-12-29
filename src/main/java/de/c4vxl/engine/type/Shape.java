package de.c4vxl.engine.type;

import de.c4vxl.engine.utils.TensorUtils;

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
}
