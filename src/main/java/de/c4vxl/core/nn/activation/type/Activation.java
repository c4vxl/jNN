package de.c4vxl.core.nn.activation.type;

import de.c4vxl.core.nn.activation.*;
import de.c4vxl.core.nn.module.Module;
import de.c4vxl.core.tensor.Tensor;

public abstract class Activation extends Module {
    public abstract <T> Tensor<T> forward(Tensor<T> input);

    public static Activation ReLU() { return new ReLU(); }
    public static Activation LeakyReLU() { return new LeakyReLU(); }
    public static Activation GELU() { return new GELU(); }
    public static Activation Sigmoid() { return new Sigmoid(); }
    public static Activation Softmax() { return new Softmax(); }
    public static Activation TanH() { return new TanH(); }
}