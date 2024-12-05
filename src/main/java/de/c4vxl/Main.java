package de.c4vxl;

import de.c4vxl.engine.data.DType;
import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.nn.Linear;
import de.c4vxl.models.LSTMForNLP;
import de.c4vxl.tokenizers.BPETokenizer;

import java.nio.charset.StandardCharsets;
import java.util.*;

public class Main {
    public static void main(String[] args) {
//        LSTMForNLP model = new LSTMForNLP(50000, 64, 64, 2);
//        System.out.println(
//                model.generate(Tensor.of(4, 3), 30, (idx, i) -> {
//                    System.out.println(i + " : " + idx);
//                }, 30)
//        );



        BPETokenizer tkn = BPETokenizer.train("Hello, how are you?", 10);
        System.out.println(
                tkn._decode(List.of(3, 33))
        );
    }
}