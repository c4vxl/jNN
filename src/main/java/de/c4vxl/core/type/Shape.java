package de.c4vxl.core.type;

import de.c4vxl.core.utils.TensorUtils;

import java.util.Arrays;

public class Shape {
    public Integer[] dimensions;

    public int size = -1;

    public Shape(Integer... shape) {
        this.dimensions = shape;
    }

    /**
     * Returns the amount of dimensions this Shape has
     */
    public int rank() { return dimensions.length; }

    /**
     * Returns the amount of data this Tensor can store
     */
    public int size() {
        if (this.size == -1)
            this.size = TensorUtils.shapeToSize(this.dimensions);
        return this.size;
    }


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
