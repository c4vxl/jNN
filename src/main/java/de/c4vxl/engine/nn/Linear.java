package de.c4vxl.engine.nn;

import de.c4vxl.engine.module.Module;
import de.c4vxl.engine.tensor.Tensor;
import de.c4vxl.engine.type.DType;

/**
 * A Linear-Layer-Module used for performing linear transformation by multiplying the input with learnable weights and adding an optional bias
 */
public class Linear extends Module {
    public Tensor<?> weight;
    public Tensor<?> bias;

    public Linear(int in_features, int out_features) { this(in_features, out_features, true); }
    public Linear(int in_features, int out_features, boolean bias) {
        this.weight = Tensor.ones(in_features, out_features).asDType(DType.DEFAULT);
        this.bias = bias ? Tensor.zeros(out_features).asDType(DType.DEFAULT) : null;
    }

    public <T> Tensor<T> forward(Tensor<T> input) {
        Tensor<T> result = input.matmul(this.weight.asDType(input.dtype));

        if (this.bias != null)
            result = result.add(this.bias.asDType(input.dtype));

        return result;
    }
}