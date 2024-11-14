package de.c4vxl.engine.nn;

import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.module.Module;

/**
 * Object for linear transformation by multiplying with weights and adding an optional bias
 *
 * @author c4vxl
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class Linear extends Module {
    public Tensor weight;
    public Tensor bias;

    public Linear(int input_features, int output_features) { this(input_features, output_features, Tensor.defaultDataType, true); }
    public Linear(int input_features, int output_features, boolean useBias) { this(input_features, output_features, Tensor.defaultDataType, useBias); }
    public Linear(int input_features, int output_features, Class<?> dtype) { this(input_features, output_features, dtype, true); }
    public Linear(int input_features, int output_features, Class<?> dtype, boolean useBias) {
        this.weight = Tensor.ones(dtype, input_features, output_features);
        this.bias = useBias ? Tensor.zeros(dtype, output_features) : null;
    }

    public <T> Tensor<T> forward(Tensor<T> x) {
        int[] shape = x.shape.clone();
        shape[shape.length - 1] = this.weight.size(1);

        // reshape bias & input if it is 3-dimensional (batch, seq_len, n_embd)
        Tensor bias = this.bias;
        if (x.is3d()) {
            x = x.reshape(x.size(0) * x.size(1), x.size(2));
            bias = bias == null ? null : bias.reshapeUnsafe(shape);
        }

        Tensor<T> result = x.matmul(this.weight);

        if (this.bias != null)
            result = result.add(bias);

        result = result.reshape(shape);

        return result;
    }
}