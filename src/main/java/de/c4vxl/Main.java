package de.c4vxl;

import de.c4vxl.engine.data.Tensor;
import de.c4vxl.models.LSTMForNLP;

public class Main {
    public static void main(String[] args) {
        LSTMForNLP model = new LSTMForNLP(6000, 128, 64, 4);
        System.out.println(
                model.generate(
                        Tensor.of(4, 32, 23, 12, 13, 7, 12, 52, 23, 123, 324, 356, 12, 3124, 238),
                        100, (next_token, i) -> {
                            System.out.println(i + ": " + next_token);
                        }, 100
                )
        );
    }
}