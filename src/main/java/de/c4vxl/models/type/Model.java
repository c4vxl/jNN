package de.c4vxl.models.type;

import de.c4vxl.core.nn.module.Module;
import de.c4vxl.core.tensor.Tensor;

import java.util.function.Function;

/**
 * A general wrapper model around a {@code Module}
 */
public class Model<T> extends Module {
    public final Module model;
    public final Function<Tensor<T>, Tensor<T>> forwardMethod;

    /**
     * Create a new model instance
     * @param model The actual module used to generate
     * @param forwardMethod The method used to generate from the module
     */
    public Model(Module model, Function<Tensor<T>, Tensor<T>> forwardMethod) {
        this.model = model;
        this.forwardMethod = forwardMethod;
    }

    /**
     * Forward through this model
     * @param inputs The inputs to the model
     */
    public final Tensor<T> forward(Tensor<T> inputs) {
        return forwardMethod.apply(inputs);
    }
}