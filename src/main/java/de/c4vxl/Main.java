package de.c4vxl;

import de.c4vxl.engine.tensor.Tensor;
import de.c4vxl.engine.utils.TensorUtils;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        System.out.println(
                Tensor.range(12, 5, 3).matmul(
                        Tensor.range(1, 12, 3, 5)
                )
        );
    }
}