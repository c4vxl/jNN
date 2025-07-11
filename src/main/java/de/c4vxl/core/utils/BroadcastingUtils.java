package de.c4vxl.core.utils;

import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * A collection of utilities used for broadcasting of two differently shaped tensors.
 * @see de.c4vxl.core.tensor.Tensor
 */
public class BroadcastingUtils {
    /**
     * Calculate the broadcasted shape between two shapes
     * @param a The first shape
     * @param b The second shape
     */
    public static Integer[] broadcastShapes(Integer[] a, Integer[] b) {
        // pad inputs to be the same size
        int length = Math.max(a.length, b.length);
        Integer[] paddedA = DataUtils.padLeft(a, 1, length, false);
        Integer[] paddedB = DataUtils.padLeft(b, 1, length, false);

        Integer[] resultShape = new Integer[length];
        for (int i = 0; i < length; i++) {
            int dimA = paddedA[i];
            int dimB = paddedB[i];

            if (dimA != 1 && dimB != 1 && dimA != dimB)
                throw new IllegalArgumentException("Shapes " + Arrays.toString(a) + " and " + Arrays.toString(b) + " are not broadcastable!");

            resultShape[i] = Math.max(dimA, dimB);
        }

        return resultShape;
    }

    /**
     * Adjust data according to broadcasted shape
     * @param data The data to process
     * @param sourceShape The source shape of the data
     * @param broadcastedShape The new shape for the data
     */
    @SuppressWarnings("unchecked")
    public static <T> T[] broadcastData(T[] data, Integer[] sourceShape, Integer[] broadcastedShape) {
        T[] broadcastedData = (T[]) Array.newInstance(data.getClass().getComponentType(), TensorUtils.shapeToSize(broadcastedShape));

        // pad source shape to be same length
        sourceShape = TensorUtils.padShapeLeft(broadcastedShape.length, true, sourceShape);

        Integer[] strides = TensorUtils.calculateStrides(sourceShape);

        for (int i = 0; i < broadcastedData.length; i++) {
            Integer[] index = TensorUtils.unravelIndex(broadcastedShape, i);

            int originalIndex = 0;
            for (int j = 0; j < index.length; j++) {
                if (sourceShape.length > j && sourceShape[j] > 1) originalIndex += index[j] * strides[j];
            }

            broadcastedData[i] = data[originalIndex];
        }

        return broadcastedData;
    }
}
