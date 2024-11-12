package de.c4vxl.engine.nn;

import de.c4vxl.engine.data.Module;
import de.c4vxl.engine.data.Tensor;

import java.util.Random;

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
        this.weight = Tensor.zeros(dtype, input_features, output_features);
        this.bias = useBias ? Tensor.zeros(dtype, output_features) : null;

        // Xavier/Glorot initialization
        // https://paperswithcode.com/method/xavier-initialization
        double limit = Math.sqrt(6.0 / (input_features + output_features));
        Random rand = new Random();
        weight.elementWise((element, index) ->
                weight.valueOf(rand.nextDouble() * 2 * limit - limit)
        );
    }

    public <T> Tensor<T> forward(Tensor<T> x) {
        x = x.matmul(this.weight);

        if (bias != null)
            x = x.add(bias);

        return x;
    }
}