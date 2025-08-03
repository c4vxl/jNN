package de.c4vxl.models;

import de.c4vxl.core.nn.activation.type.ActivationFunction;
import de.c4vxl.core.nn.module.Module;
import de.c4vxl.core.nn.Linear;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Apply a multi-layer short-term memory (LSTM) RNN to an input sequence.
 * <a href="https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html">More information</a>
 */
public class LSTM extends Module {
    /**
     * A single LSTM Cell
     * <a href="https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html">More information</a>
     */
    public class LSTMCell extends Module {
        private int hidden_size;
        public Linear ih, hh;

        public LSTMCell(int input_size, int hidden_size, boolean bias, DType<?> dtype) {
            this.ih = new Linear(input_size, 4 * hidden_size, bias, dtype);
            this.hh = new Linear(hidden_size, 4 * hidden_size, bias, dtype);
            this.hidden_size = hidden_size;
        }

        public <T> List<Tensor<T>> forward(Tensor<T> input, Tensor<T> h_prev, Tensor<T> c_prev) {
            if (input.dim() != 1 && input.dim() != 2)
                throw new IllegalArgumentException("Expected input to be 1D or 2D. But got " + input.dim() + "D Tensor!");

            // prepare input
            boolean is_batched = input.dim() == 2;
            if (!is_batched)
                input = input.unsqueeze(0);

            // prepare h_prev and c_prev
            if (h_prev == null || c_prev == null) {
                h_prev = Tensor.zeros(input.size(0), this.hidden_size).asDType(input.dtype);
                c_prev = h_prev.clone();
            } else if (!is_batched) {
                h_prev = h_prev.unsqueeze(0);
                c_prev = c_prev.unsqueeze(0);
            }

            // calculate gates
            Tensor<T>[] gates = TensorUtils.chunk(this.ih.forward(input)
                    .add(this.hh.forward(h_prev)), -1, this.hidden_size);
            Tensor<T> i = ActivationFunction.Sigmoid(gates[0]);
            Tensor<T> f = ActivationFunction.Sigmoid(gates[1]);
            Tensor<T> g = gates[2].tanh();
            Tensor<T> o = ActivationFunction.Sigmoid(gates[3]);

            // calculate hidden and cell state
            Tensor<T> c_next = f.mul(c_prev).add(i.mul(g));
            Tensor<T> h_next = o.mul(c_next.tanh());
//            if (!is_batched) {
//                c_next = c_next.squeeze(0);
//                h_next = h_next.squeeze(0);
//            }

            return List.of(h_next, c_next);
        }
    }

    public static class LSTMOutput<T> {
        public Tensor<T> result;
        public List<Tensor<Double>> hx;

        public LSTMOutput(Tensor<T> result, Tensor<Double> h, Tensor<Double> c) {
            this.result = result;
            this.hx = List.of(h, c);
        }

        @Override
        public String toString() {
            return "LSTMOutput{" +
                    "result=" + result +
                    ", hx=" + hx +
                    '}';
        }
    }

    public ArrayList<LSTMCell> cells;
    public int num_layers, hidden_size, input_size;
    protected ArrayList<Tensor<Double>> last_hx = null;

    public LSTM(int input_size, int hidden_size, int num_layers, boolean bias, DType<?> dtype) {
        this.num_layers = num_layers;
        this.hidden_size = hidden_size;
        this.input_size = input_size;

        this.cells = new ArrayList<>(List.of(new LSTMCell(input_size, hidden_size, bias, dtype)));
        for (int i = 0; i < num_layers; i++) {
            this.cells.add(new LSTMCell(hidden_size, hidden_size, bias, dtype));
        }
    }

    public <T> Tensor<T> forward(Tensor<T> input) {
        LSTMOutput<T> out = this.forward(input, null);
        this.last_hx = new ArrayList<>(out.hx);
        return out.result;
    }
    @SuppressWarnings("unchecked")
    public <T> LSTMOutput<T> forward(Tensor<T> input, List<Tensor<Double>> hidden) {
        Tensor<Double> x = input.asDouble();

        // batch x if necessary
        boolean is_batched = x.dim() == 3;
        if (!is_batched)
            x = x.unsqueeze(0); // batch, seq_length, n_embd

        // extract dimension sizes
        int B = x.size(0); // batch
        int T = x.size(1); // seq_length

        // generate/load h and c states
        ArrayList<Tensor<Double>> h_t = new ArrayList<>(), c_t = new ArrayList<>();
        if (hidden == null) {
            for (int i = 0; i < this.num_layers; i++) {
                Tensor<Double> zeros = Tensor.zeros(B, this.hidden_size).asDouble();
                h_t.add(zeros.clone());
                c_t.add(zeros.clone());
            }
        } else {
            for (int i = 0; i < this.num_layers; i++) {
                h_t.add(hidden.get(0).get(i));
                c_t.add(hidden.get(1).get(i));
            }
        }

        Tensor<T>[] outputs = new Tensor[T];
        for (int t = 0; t < T; t++) {
            Tensor<Double> x_t = x.get(null, t, null); // extract the t-th time step

            // pass "x_t" through cell
            for (int i = 0; i < this.num_layers; i++) {
                List<Tensor<Double>> output = this.cells.get(i).forward(x_t, h_t.get(i), c_t.get(i));
                h_t.set(i, output.get(0));
                c_t.set(i, output.get(1));
                x_t = h_t.get(i);
            }

            outputs[t] = h_t.getLast().asDType(input.dtype);
        }

        return new LSTMOutput<>(TensorUtils.stack(1, outputs),
                TensorUtils.stack(0, h_t.toArray(Tensor[]::new)), // concatenate h back to tensor
                TensorUtils.stack(0, c_t.toArray(Tensor[]::new))); // concatenate c back to tensor
    }
}