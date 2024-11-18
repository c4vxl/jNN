package de.c4vxl;

import de.c4vxl.engine.data.Tensor;
import de.c4vxl.models.GPT2StyleModel;

public class Main {
    public static void main(String[] args) {
        // load or initialize random model
        GPT2StyleModel model = (GPT2StyleModel) GPT2StyleModel.load("model.mdl");
        model = model == null ? new GPT2StyleModel(180, 6, 6, 50, 100, true) : model;

        System.out.println(model.generate(
                Tensor.of(3., 1., 4., 2., 6.) // sample tokens
        ));
    }
}