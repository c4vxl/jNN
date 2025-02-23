package de.c4vxl.engine.activation;

import de.c4vxl.engine.tensor.Tensor;
import de.c4vxl.engine.type.DType;
import de.c4vxl.engine.utils.TensorUtils;

public class ActivationFunction {
    /**
     * Apply element-wise Rectified linear unit. Relu is defined as: `relu(x) = max(0, x)`
     * @param input The input tensor
     */
    public static <T> Tensor<T> ReLU(Tensor<T> input) { return input.clip(0, Double.MAX_VALUE); }

    /**
     * Apply element-wise Gaussian Error Linear Unit.
     * (See `<a href="https://pytorch.org/docs/stable/generated/torch.nn.GELU.html">Pytorch docs</a>` for formula)
     * @param input The input tensor
     */
    public static <T> Tensor<T> GELU(Tensor<T> input) {
        return TensorUtils.elementWise(input, (a, i) -> {
            double x = DType.DOUBLE.parse(a);
            return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
        });
    }

    /**
     * Perform element-wise Sigmoid
     * Sigmoid is defined as `sigmoid(x) = 1 / (1 + e^-x)`
     * @param input The tensor
     */
    public static <T> Tensor<T> Sigmoid(Tensor<T> input) {
        return TensorUtils.elementWise(input, (a, i) -> 1 / (1 + Math.exp(-1 * DType.DOUBLE.parse(a))));
    }

    /**
     * Apply Hyperbolic Tangent (Tanh) element-wise
     * Tanh is defined as `tanh(x) = (exp(x)−exp(−x))/exp(x)+exp(−x)`
     * @param input The input Tensor
     */
    public static <T> Tensor<T> tanh(Tensor<T> input) {
        return TensorUtils.elementWise(input, (a, i) -> Math.tanh(DType.DOUBLE.parse(a)));
    }

    /**
     * Apply element wise softmax over the last dimension (-1)
     * A softmax turns a list of values into a probability distribution which sums up to 1.
     * @param input The input tensor
     */
    public static <T> Tensor<T> Softmax(Tensor<T> input) { return ActivationFunction.Softmax(input, 1); }

    /**
     * Apply element wise softmax over the last dimension (-1)
     * A softmax turns a list of values into a probability distribution which sums up to 1.
     * @param input The input tensor
     * @param temperature The significance smaller values should receive
     */
    public static <T> Tensor<T> Softmax(Tensor<T> input, double temperature) {
        return ActivationFunction.Softmax(input, temperature, -1);
    }

    /**
     * Apply element wise softmax over a specified dimension
     * A softmax turns a list of values into a probability distribution which sums up to 1.
     * @param input The input tensor
     * @param temperature The significance smaller values should receive
     * @param dim The dimension to apply the Softmax over
     */
    public static <T> Tensor<T> Softmax(Tensor<T> input, double temperature, int dim) {
        // scale by temperature
        Tensor<T> scaledInput = input.div(input.dtype.parse(temperature));

        // apply log-sum-exp (LSE) for stabilization
        Tensor<T> stabilizedInput = scaledInput.sub(TensorUtils.reduceAlongDimension(scaledInput, dim, Tensor::max, true));

        // apply the actual softmax
        Tensor<T> expInput = stabilizedInput.exp();
        return expInput.div(expInput.sum(dim, true));
    }
}