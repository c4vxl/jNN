package de.c4vxl;

import de.c4vxl.engine.data.DType;
import de.c4vxl.engine.data.Tensor;
import de.c4vxl.models.LSTM;

public class Main {
    public static void main(String[] args) {
        LSTM lstmModel = LSTM.load("mod.mdl");
        assert lstmModel != null;

        Tensor<Double> input = Tensor.range(DType.DOUBLE, 10).unsqueeze(0);

        LSTM.LSTMOutput<Double> lastOut = null;
        for (int i = 0; i < 50; i++) {
            if (lastOut == null)
                lastOut = lstmModel.forward(input);
            else
                lastOut = lstmModel.forward(input, lastOut.h, lastOut.c);
        }

        System.out.println(lastOut.output);
    }
}