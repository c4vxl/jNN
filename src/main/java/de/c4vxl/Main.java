package de.c4vxl;

import de.c4vxl.engine.activation.Activation;
import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.data.TensorUtils;
import de.c4vxl.models.GPT2StyleModel;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        Tensor<Double> input_ids = Tensor.of(1., 4.).unsqueeze(0);

        GPT2StyleModel model = new GPT2StyleModel(4, 2, 2, 10, 30, true);
        model = (GPT2StyleModel) model.load("model.xml");

        System.out.println(model.forward(input_ids));
    }

    public static Tensor<Integer> generate(GPT2StyleModel model, Tensor<Integer> input_ids, int max_new_tokens, int eos_token_id) {
        if (!input_ids.is1d())
            throw new IllegalArgumentException("input_ids must be 1-dimensional!");

        input_ids = input_ids.unsqueeze(0); // 1, seq_len

        for (int i = 0; i < max_new_tokens; i++) {
            System.out.println(input_ids);
            Tensor<Integer> logits = model.forward(input_ids);
            logits = TensorUtils.slice(logits, new int[]{0}).squeeze(0);
            logits = Activation.Softmax(logits);

            Tensor<Integer> next_token = Tensor.of(Arrays.stream(logits.data).toList().indexOf(logits.max()));

            if (next_token.item() == eos_token_id)
                break;

            input_ids = input_ids.concatenate(next_token);
        }

        return input_ids;
    }
}