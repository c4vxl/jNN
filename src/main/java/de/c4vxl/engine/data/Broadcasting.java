package de.c4vxl.engine.data;

public class Broadcasting {
    /**
     * Determines the broadcasted shape for two shapes, or returns null if they are not broadcastable.
     * @param shapeA First shape
     * @param shapeB Second shape
     */
    public static int[] getBroadcastedShape(int[] shapeA, int[] shapeB) {
        int maxLength = Math.max(shapeA.length, shapeB.length);
        int[] resultShape = new int[maxLength];

        for (int i = 0; i < maxLength; i++) {
            int dimA = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : 1;
            int dimB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1;

            if (dimA != dimB && dimA != 1 && dimB != 1) {
                return null;  // Shapes are not broadcastable
            }
            resultShape[maxLength - 1 - i] = Math.max(dimA, dimB);
        }

        return resultShape;
    }

    /**
     * Broadcast a Tensor to a shape
     * @param tensor The Tensor
     * @param shape The shape to broadcast to
     */
    public static <T> Tensor<T> broadcast(Tensor<T> tensor, int[] shape) {
        Tensor<T> t = tensor.clone();
        return t.reshapeUnsafe(getBroadcastedShape(tensor.shape, shape));
    }

    /**
     * Broadcast a Tensor to a shape
     * @param tensor The Tensor
     * @param b The Tensor to broadcast to
     */
    public static <T> Tensor<T> broadcast(Tensor<T> tensor, Tensor<?> b) {
        return broadcast(tensor, b.shape);
    }

    /**
     * Checks if a Tensor is broadcastable to a shape
     * @param tensor The Tensor
     * @param shape The Shape
     */
    public static boolean isBroadcastable(Tensor<?> tensor, int[] shape) {
        return getBroadcastedShape(tensor.shape, shape) != null;
    }
}