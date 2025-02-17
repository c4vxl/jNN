package de.c4vxl.models;

import de.c4vxl.engine.activation.Activation;
import de.c4vxl.engine.activation.ActivationFunction;
import de.c4vxl.engine.module.Module;
import de.c4vxl.engine.nn.Embedding;
import de.c4vxl.engine.nn.LayerNorm;
import de.c4vxl.engine.nn.Linear;
import de.c4vxl.engine.nn.Sequence;
import de.c4vxl.engine.tensor.Tensor;
import de.c4vxl.engine.utils.TensorUtils;
import de.c4vxl.models.type.TextGenerationModel;

import java.util.ArrayList;

public class Transformer extends TextGenerationModel {
    public static class CausalSelfAttention extends Module {
        public int n_embd, n_head;
        public Linear c_attn, c_proj;

        public CausalSelfAttention(int n_embd, int n_head, int block_size, boolean bias) {
            this.n_head = n_head;
            this.n_embd = n_embd;

            assert n_embd % n_head == 0;

            this.c_attn = new Linear(n_embd, 3 * n_embd, bias);
            this.c_proj = new Linear(n_embd, n_embd, bias);
        }

        public <T> Tensor<T> forward(Tensor<T> x) {
            int B = x.size(0), T = x.size(1), C = x.size(2); // batch, sequence length, n_embd

            // calculate q, k and v matrices
            Tensor<T>[] chunks = TensorUtils.chunk(this.c_attn.forward(x), 2, C);
            Tensor<T> k = chunks[0], q = chunks[1], v = chunks[2];

            // split into heads
            k = k.reshape(B, T, this.n_head, C / this.n_head).transpose(1, 2); // B, nh, T, hs
            q = q.reshape(B, T, this.n_head, C / this.n_head).transpose(1, 2); // B, nh, T, hs
            v = v.reshape(B, T, this.n_head, C / this.n_head).transpose(1, 2); // B, nh, T, hs

            // calculate qk value
            Tensor<T> qk = (q.matmul(k.transpose(-2, -1)))
                    .mul(k.dtype.parse(1 / Math.sqrt(k.size(-1))));

            // apply mask
            qk = TensorUtils.maskedFill(qk, TensorUtils.tril(Tensor.ones(T, T), 0), 0., Double.NEGATIVE_INFINITY);

            // apply softmax and multiply by v
            Tensor<T> qkv = ActivationFunction.Softmax(qk).matmul(v);

            // reassemble heads
            return qkv.transpose(1, 2).reshape(B, T, C);
        }
    }

    public static class Block extends Module {
        public CausalSelfAttention attn;
        public Sequence mlp;
        public LayerNorm ln_1, ln_2;

        public Block(int n_embd, int n_head, int block_size, boolean bias) {
            this.ln_1 = new LayerNorm(n_embd);
            this.attn = new CausalSelfAttention(n_embd, n_head, block_size, bias);
            this.ln_2 = new LayerNorm(n_embd);
            this.mlp = new Sequence(
                    new Linear(n_embd, n_embd * 4, bias),
                    Activation.GELU(),
                    new Linear(4 * n_embd, n_embd)
            );
        }

        @SuppressWarnings("unchecked")
        public <T> Tensor<T> forward(Tensor<T> x) {
            x = x.add(this.attn.forward(this.ln_1.forward(x)));
            x = x.add((Tensor<T>) this.mlp.forward(this.ln_2.forward(x)));
            return x;
        }
    }

    public int block_size;
    public Embedding wte, wpe;
    public LayerNorm ln_f;
    public Linear lm_head;
    public ArrayList<Block> heads = new ArrayList<>();

    public Transformer(int n_embd, int n_head, int n_layer, int block_size, int vocab_size, boolean bias) {
        this.block_size = block_size;

        // embeddings
        this.wte = new Embedding(vocab_size, n_embd);
        this.wpe = new Embedding(block_size, n_embd);

        // layer norm
        this.ln_f = new LayerNorm(n_embd);

        // blocks
        for (int i = 0; i < n_layer; i++)
            this.heads.add(new Block(n_embd, n_head, block_size, bias));

        // output projection
        this.lm_head = new Linear(n_embd, vocab_size, false);
    }

    @Override
    public <T extends Number> Tensor<T> forward(Tensor<T> input) {
        Tensor<Double> idx = input.asDouble();

        int T = idx.size(1); // sequence length

        assert T <= this.block_size: "Sequence too long!";

        // embedded tokens
        // shape: b, t, n_embd
        Tensor<Double> x = this.wte.forward(idx)
                .add(this.wpe.forward(Tensor.range(T).asDouble()));

        for (Block block : this.heads)
            x = block.forward(x);

        x = this.ln_f.forward(x);

        Tensor<Double> logits = this.lm_head.forward(x); // b, t, voc_size

        return logits.asDType(input.dtype);
    }
}