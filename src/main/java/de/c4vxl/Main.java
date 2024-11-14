package de.c4vxl;

import de.c4vxl.engine.data.Tensor;
import de.c4vxl.models.GPT2LMModel;

public class Main {
    public static void main(String[] args) {
        Tensor<Float> input_ids = Tensor.of(1F, 4F).unsqueeze(0);

        GPT2LMModel model = new GPT2LMModel(4, 2, 2, 10, 30, true);

        model.load("model_params.xml");

        Tensor<Float> output = model.forward(input_ids);

        System.out.println(output);
    }
}