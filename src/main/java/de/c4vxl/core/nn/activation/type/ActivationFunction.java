package de.c4vxl.core.nn.activation.type;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.*;
import de.c4vxl.core.utils.TensorUtils;

/**
 * This class contains a collection of activation functions
 */
public class ActivationFunction {
    /**
     * Apply element-wise Rectified linear unit. Relu is defined as: `relu(x) = max(0, x)`
     * @param input The input tensor
     * @see de.c4vxl.core.tensor.operation.ReLUOperation
     */
    public static <T> Tensor<T> ReLU(Tensor<T> input) { return new ReLUOperation<>(input).forward(); }

    /**
     * Apply element-wise Leaky Rectified linear unit.
     * @param input The input tensor
     * @param alpha The alpha
     * @see de.c4vxl.core.tensor.operation.LeakyReLUOperation
     */
    public static <T> Tensor<T> LeakyReLU(Tensor<T> input, double alpha) { return new LeakyReLUOperation<>(input, alpha).forward(); }

    /**
     * Apply element-wise Gaussian Error Linear Unit.
     * (See `<a href="https://pytorch.org/docs/stable/generated/torch.nn.GELU.html">Pytorch docs</a>` for formula)
     * @param input The input tensor
     * @see de.c4vxl.core.tensor.operation.GELUOperation
     */
    public static <T> Tensor<T> GELU(Tensor<T> input) { return new GELUOperation<>(input).forward(); }

    /**
     * Perform element-wise Sigmoid
     * Sigmoid is defined as `sigmoid(x) = 1 / (1 + e^-x)`
     * @param input The tensor
     * @see de.c4vxl.core.tensor.operation.SigmoidOperation
     */
    public static <T> Tensor<T> Sigmoid(Tensor<T> input) { return new SigmoidOperation<>(input).forward(); }

    /**
     * Apply element wise softmax over the last dimension (-1)
     * A softmax turns a list of values into a probability distribution which sums up to 1.
     * @param input The input tensor
     * @see ActivationFunction#Softmax
     */
    public static <T> Tensor<T> Softmax(Tensor<T> input) { return ActivationFunction.Softmax(input, 1); }

    /**
     * Apply element wise softmax over the last dimension (-1)
     * A softmax turns a list of values into a probability distribution which sums up to 1.
     * @param input The input tensor
     * @param temperature The significance smaller values should receive
     * @see ActivationFunction#Softmax
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