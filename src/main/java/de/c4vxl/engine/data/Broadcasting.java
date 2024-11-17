package de.c4vxl.engine.data;

import java.util.Arrays;

/**
 * Helper class implementing Broadcasting rules and utilities!
 *
 * @author c4vxl
 */
public class Broadcasting {
    /**
     * Broadcast two shapes into a shared shape
     * @param shapeA Shape a
     * @param shapeB Shape b
     */
    public static int[] broadcastShapes(int[] shapeA, int[] shapeB) {
        int lenA = shapeA.length;
        int lenB = shapeB.length;
        int maxLength = Math.max(lenA, lenB);

        // align dimensions by padding with 1s
        int[] paddedA = padLeft(shapeA, maxLength);
        int[] paddedB = padLeft(shapeB, maxLength);

        int[] resultShape = new int[maxLength];
        for (int i = 0; i < maxLength; i++) {
            int dimA = paddedA[i];
            int dimB = paddedB[i];

            if (dimA != 1 && dimB != 1 && dimA != dimB) {
                throw new IllegalArgumentException("Shapes cannot be broadcast: " +
                        Arrays.toString(shapeA) + " and " + Arrays.toString(shapeB));
            }

            resultShape[i] = Math.max(dimA, dimB);
        }

        return resultShape;
    }

    /**
     * Adds 1s to the left of the shape until a specified target length is reached
     * @param shape The Shape to pad
     * @param targetLength The target length of the shape
     */
    private static int[] padLeft(int[] shape, int targetLength) {
        int[] paddedShape = new int[targetLength];
        int offset = targetLength - shape.length;

        for (int i = 0; i < targetLength; i++) {
            paddedShape[i] = (i < offset) ? 1 : shape[i - offset];
        }

        return paddedShape;
    }

    /**
     * Compute broadcasted indicies for a source to a target
     */
    public static int[] getBroadcastedIndices(int batchIdx, int[] tensorShape, int[] broadcastedShape) {
        int[] indices = new int[tensorShape.length];
        int offset = broadcastedShape.length - tensorShape.length;

        for (int i = 0; i < tensorShape.length; i++) {
            if (tensorShape[i] == 1)
                indices[i] = 0;  // singleton dimensions repeat the value along the expanded dimensionselse
            else
                indices[i] = (batchIdx / TensorUtils.computeStride(broadcastedShape, offset + i)) % tensorShape[i];
        }

        return indices;
    }

    /**
     * Broadcast Tensor to a specific Shape
     * @param tensor The Tensor
     * @param targetShape The target shape to broadcast to
     */
    public static <T> Tensor<T> broadcastTo(Tensor<T> tensor, int[] targetShape) {
        int[] broadcastedShape = broadcastShapes(tensor.shape, targetShape);
        Tensor<T> result = new Tensor<>(tensor.dtype, broadcastedShape);

        for (int i = 0; i < result.data.length; i++) {
            int[] broadcastedIndices = getBroadcastedIndices(i, tensor.shape, broadcastedShape);
            int sourceIndex = TensorUtils.computeBatchOffset(broadcastedIndices, tensor.shape);

            // System.out.println("Flat index " + i + " => Broadcasted indices: " + Arrays.toString(broadcastedIndices)
            //         + " => Source index: " + sourceIndex); // debugging

            result.data[i] = tensor.data[sourceIndex];
        }

        return result;
    }
}