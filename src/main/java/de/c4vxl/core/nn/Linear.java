package de.c4vxl.core.nn;

import de.c4vxl.core.nn.module.Module;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.type.DType;

/**
 * A Linear-Layer-Module used for performing linear transformation by multiplying the input with learnable weights and adding an optional bias
 */
public class Linear extends Module {
    public Tensor<?> weight;
    public Tensor<?> bias;

    public Linear(int in_features, int out_features) { this(in_features, out_features, true); }
    public Linear(int in_features, int out_features, boolean bias) { this(in_features, out_features, bias, DType.DEFAULT); }
    public Linear(int in_features, int out_features, boolean bias, DType<?> dtype) {
        this.weight = Tensor.ones(in_features, out_features).asDType(dtype);
        this.bias = bias ? Tensor.zeros(out_features).asDType(dtype) : null;
    }

    public <T> Tensor<T> forward(Tensor<T> input) {
        Tensor<T> result = input.matmul(this.weight.asDType(input.dtype));

        if (this.bias != null)
            result = result.add(this.bias.asDType(input.dtype));

        return result;
    }
}