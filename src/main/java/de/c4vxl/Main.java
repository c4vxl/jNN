package de.c4vxl;

import de.c4vxl.engine.tensor.Tensor;

public class Main {
    public static void main(String[] args) {
        Tensor<Integer> a = Tensor.range(5, 3, 2);
        System.out.println(a);
        Tensor<Integer> slice = a.get(0, 0);
        System.out.println(slice);

        System.out.println(
                a.set(1, 0, 0, 0)
        );
    }
}