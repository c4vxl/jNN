package de.c4vxl.engine.utils;

import java.util.Arrays;

public class TensorUtils {
    /**
     * Returns the size of the shape if it was 1d
     * @param shape The shape
     */
    public static int shapeToSize(int... shape) { return Arrays.stream(shape).reduce(1, (a, b) -> a * b); }
}
