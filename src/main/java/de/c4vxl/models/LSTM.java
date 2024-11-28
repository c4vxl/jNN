package de.c4vxl.models;

import de.c4vxl.engine.activation.Activation;
import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.data.TensorUtils;
import de.c4vxl.engine.module.Module;
import de.c4vxl.engine.nn.Linear;

import java.util.ArrayList;
import java.util.List;

public class LSTM extends Module {
    public static class LSTMOutput<T extends Number> {
        public List<Tensor<T>> h;
        public List<Tensor<T>> c;
        public Tensor<T> output;

        public LSTMOutput(Tensor<T> output, List<Tensor<T>> h, List<Tensor<T>> c) {
            this.h = h;
            this.c = c;
            this.output = output;
        }
    }

    public static class LSTMCell extends Module {
        public Linear ih; // forget gate
        public Linear hh; // input gate
        private final int hidden_size;

        public LSTMCell(int input_size, int hidden_size, boolean bias) {
            this.hidden_size = hidden_size;

            this.ih = new Linear(input_size, 4 * hidden_size, bias);
            this.hh = new Linear(hidden_size, 4 * hidden_size, bias);
        }

        public <T> List<Tensor<T>> forward(Tensor<T> x, Tensor<T> h_prev, Tensor<T> c_prev) {
            boolean is_batched = x.rank() == 2;
            if (!is_batched) {
                x = x.unsqueeze(0);
                h_prev = h_prev.unsqueeze(0);
                c_prev = c_prev.unsqueeze(0);
            }

            Tensor<T>[] gates = TensorUtils.chunk(
                    this.ih.forward(x).add(this.hh.forward(h_prev)) // calculate all gates together
                    , this.hidden_size); // chunk into single gates

            Tensor<T> i = Activation.Sigmoid(gates[0]);
            Tensor<T> f = Activation.Sigmoid(gates[1]);
            Tensor<T> g = Activation.tanh(gates[2]);
            Tensor<T> o = Activation.Sigmoid(gates[3]);

            Tensor<T> c_next = f.mul(c_prev).add(i.mul(g)); // f * c_prev + i * g
            Tensor<T> h_next = o.mul(Activation.tanh(c_next)); // o * tanh(c_next)

            if (!is_batched) {
                c_next = c_next.squeeze(0);
                h_next = h_next.squeeze(0);
            }

            return List.of(h_next, c_next);
        }
    }

    private final int hidden_size;
    private final int num_layers;
    private final int proj_size;
    public List<Module> cells;

    public LSTM(int input_size, int hidden_size) { this(input_size, hidden_size, 1); }
    public LSTM(int input_size, int hidden_size, int num_layers) { this(input_size, hidden_size, num_layers, 0, true); }
    public LSTM(int input_size, int hidden_size, int num_layers, int proj_size, boolean bias) {
        this.hidden_size = hidden_size;
        this.num_layers = num_layers;
        this.proj_size = proj_size;

        this.cells = new ArrayList<>(List.of(new LSTMCell(input_size, hidden_size, bias)));
        for (int i = 0; i < num_layers - 1; i++) {
            this.cells.add(new LSTMCell(hidden_size, hidden_size, bias));
            if (proj_size > 0)
                this.cells.add(new Linear(hidden_size, proj_size, bias));
        }
    }

    public <T extends Number> LSTMOutput<T> forward(Tensor<T> input) { return forward(input, null, null); }
    @SuppressWarnings("unchecked")
    public <T extends Number> LSTMOutput<T> forward(Tensor<T> input, List<Tensor<T>> h, List<Tensor<T>> c) {
        boolean is_batched = input.rank() == 3;
        if (!is_batched)
            input = input.unsqueeze(1);

        int T = input.size(0);
        int B = input.size(1);

        // initialize h and c
        if (h == null || c == null) {
            h = new ArrayList<>();
            c = new ArrayList<>();

            for (int layer = 0; layer < num_layers; layer++) {
                h.add(Tensor.zeros(input.dtype, B, (proj_size == 0) ? hidden_size : proj_size)); // b, hidden_size/proj_size
                c.add(Tensor.zeros(input.dtype, B, this.hidden_size)); // b, hidden_size
            }
        }

        List<Tensor<T>> outputs = new ArrayList<>();

        // pass through cells
        for (int t = 0; t < T; t++) {
            Tensor<T> x = TensorUtils.slice(input, new int[]{t});

            for (int layer = 0; layer < num_layers; layer++) {
                LSTMCell cell = (LSTMCell) cells.get(layer);
                Tensor<T> hPrev = h.get(layer);
                Tensor<T> cPrev = c.get(layer);

                // update h and c using LSTMCell
                List<Tensor<T>> hcNext = cell.forward(x, hPrev, cPrev);
                Tensor<T> hNext = hcNext.get(0);
                Tensor<T> cNext = hcNext.get(1);

                h.set(layer, hNext);
                c.set(layer, cNext);

                if (proj_size > 0 && layer % 2 == 1) { // apply projection
                    Linear projection = (Linear) cells.get(layer + 1);
                    x = projection.forward(hNext);
                } else
                    x = hNext; // use hidden state as input for next layer
            }

            outputs.add(x);
        }

        Tensor<T> result = TensorUtils.stack(0, outputs.toArray(Tensor[]::new));

        // remove batch again
        if (!is_batched) {
            result = result.squeeze(1);
            h.replaceAll(tTensor -> tTensor.squeeze(0));
            c.replaceAll(tTensor -> tTensor.squeeze(0));
        }

        return new LSTMOutput<>(result, h, c);
    }
}