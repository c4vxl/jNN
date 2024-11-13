package de.c4vxl;

import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.data.TensorUtils;

public class Main {
    public static void main(String[] args) {
        Tensor<Integer> test = Tensor.ones(3, 10, 10, 1, 1, 10, 1, 30);

        System.out.println(test);
        System.out.println(TensorUtils.withoutBatchDim(test));
    }
}