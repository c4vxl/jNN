package de.c4vxl.engine.data;

/**
 * Collection of multiple loss functions
 */
public class LossFunction {
    // Mean Squared Error (MSE)
    public static <T> Tensor<T> meanSquaredError(Tensor<T> prediction, Tensor<T> label) {
        return prediction.sub(label).pow(2.0).sum(0);
    }

    // Cross-Entropy Loss
    public static Tensor<Double> crossEntropyLoss(Tensor<Double> prediction, Tensor<Double> label) { return crossEntropyLoss(prediction, label, 1); }
    public static Tensor<Double> crossEntropyLoss(Tensor<Double> prediction, Tensor<Double> label, int axis) {
        prediction = Activation.Softmax(prediction);

        // clip to avoid log(0)
        Tensor<Double> clippedPredictions = prediction.clip(1e-9, 1 - 1e-9);

        // calculate -y * log(y_hat)
        Tensor<Double> logPredictions = clippedPredictions.log();  // Element-wise log
        Tensor<Double> elementWiseLoss = label.mul(logPredictions).mul(-1.0);

        return elementWiseLoss.sum(axis);
    }
}
