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
        double bound = 1. / Math.sqrt(in_features);
        this.weight = Tensor.random(dtype, -bound, bound, in_features, out_features).asDType(dtype);
        this.bias = bias ? Tensor.random(dtype, -bound, bound, out_features).asDType(dtype) : null;
    }

    public <T> Tensor<T> forward(Tensor<T> input) {
        boolean wasUnsqueezed = false;
        if (input.dim() == 1) {
            wasUnsqueezed = true;
            input = input.unsqueeze(0);
        }

        Tensor<T> result = input.matmul(this.weight.asDType(input.dtype));

        if (this.bias != null)
            result = result.add(this.bias.asDType(input.dtype));

        if (wasUnsqueezed)
            return result.squeeze(0);

        return result;
    }
}