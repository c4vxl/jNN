package de.c4vxl.engine.nn;

import de.c4vxl.engine.data.DType;
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

    public Linear(int input_features, int output_features) { this(input_features, output_features, DType.DEFAULT, true); }
    public Linear(int input_features, int output_features, boolean useBias) { this(input_features, output_features, DType.DEFAULT, useBias); }
    public Linear(int input_features, int output_features, Class<?> dtype) { this(input_features, output_features, dtype, true); }
    public Linear(int input_features, int output_features, Class<?> dtype, boolean useBias) {
        this.weight = Tensor.ones(dtype, input_features, output_features);
        this.bias = useBias ? Tensor.zeros(dtype, output_features) : null;
    }

    public <T> Tensor<T> forward(Tensor<T> x) {
        Tensor<T> result;

        try {
            result = x.matmul(this.weight);
        } catch (Exception e) {
            result = x.matmul(this.weight.transpose());
        }

        if (this.bias != null)
            result = result.add(this.bias);

        return result;
    }
}