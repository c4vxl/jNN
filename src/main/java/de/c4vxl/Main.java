package de.c4vxl;

import de.c4vxl.engine.nn.LayerNorm;
import de.c4vxl.engine.data.Tensor;

public class Main {
    public static void main(String[] args) {
        LayerNorm norm = new LayerNorm(3);

        Tensor<Float> x = Tensor.of(5.0F, 10.0F, 19.0F);

        System.out.println(norm.forward(x));
    }
}