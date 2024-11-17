package de.c4vxl.models;

import de.c4vxl.engine.activation.Activation;
import de.c4vxl.engine.activation.GELU;
import de.c4vxl.engine.data.DType;
import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.data.TensorUtils;
import de.c4vxl.engine.module.Module;
import de.c4vxl.engine.nn.Embedding;
import de.c4vxl.engine.nn.LayerNorm;
import de.c4vxl.engine.nn.Linear;
import de.c4vxl.engine.nn.Sequence;

import java.util.ArrayList;
import java.util.List;

public class GPT2StyleModel extends Module {
    public static class CausalSelfAttention extends Module {
        private int n_head;

        public Linear q_proj;
        public Linear k_proj;
        public Linear v_proj;
        public Linear c_proj;

        public CausalSelfAttention(int n_embd, int n_head, int block_size, boolean bias) {
            this.n_head = n_head;

            this.q_proj = new Linear(n_embd, n_embd, bias);
            this.k_proj = new Linear(n_embd, n_embd, bias);
            this.v_proj = new Linear(n_embd, n_embd, bias);
            this.c_proj = new Linear(n_embd, n_embd, bias);
        }

        public <T> Tensor<T> forward(Tensor<T> x) {
            int B = x.size(0); // batch dim
            int T = x.size(1); // time dim
            int C = x.size(2); // n_embd

            Tensor<Integer> mask = TensorUtils.tril(Tensor.ones(T, T)).reshape(1, 1, T, T);

            Tensor<T> q = this.q_proj.forward(x).reshape(B, T, this.n_head, C / this.n_head).transpose(0, 2, 1, 3);                               // b, nh, t, hs
            Tensor<T> k = this.k_proj.forward(x).reshape(B, T, this.n_head, C / this.n_head).transpose(0, 2, 1, 3).transpose(0, 1, 3, 2);  // b, nh, hs, t
            Tensor<T> v = this.v_proj.forward(x).reshape(B, T, this.n_head, C / this.n_head).transpose(0, 2, 1, 3);                               // b, nh, t, hs

            Tensor<T> qk = q.matmul(k) // q @ k
                    .mul(k.valueOf(1.0 / Math.sqrt(k.size(-1)))); // q @ k / sqrt(dₖ)

            qk = TensorUtils.maskedFill(qk, mask, 0, qk.valueOf("-Infinity")); // apply mask

            Tensor<T> qkv = Activation.Softmax(qk).matmul(v); // multiply with v; shape: b, nh, t, hs

            // re-assemble all heads
            Tensor<T> y = qkv.transpose(0, 2, 1, 3).reshape(B, T, C); // b, seq_len, n_embd
            y = this.c_proj.forward(y); // b, t, n_embd

            return y;
        }
    }

    public static class Block extends Module {
        public CausalSelfAttention attn;
        public Sequence mlp;
        public LayerNorm ln_1;
        public LayerNorm ln_2;

        public Block(int n_embd, int n_head, int block_size, boolean bias) {
            this.ln_1 = new LayerNorm(n_embd);
            this.ln_2 = new LayerNorm(n_embd);

            this.attn = new CausalSelfAttention(n_embd, n_head, block_size, bias);
            this.mlp = new Sequence(
                    new Linear(n_embd, n_embd * 4, bias),
                    new GELU(),
                    new Linear(4 * n_embd, n_embd, bias)
            );
        }

        public <T> Tensor<T> forward(Tensor<T> x) {
            x = x.add(this.attn.forward(this.ln_1.forward(x))); // x + attn(x)
            x = x.add(this.mlp.forward(this.ln_2.forward(x)));  // x + mlp(x)

            return x;
        }
    }

    private int block_size;

    public Embedding wte;
    public Embedding wpe;
    public List<Block> heads;
    public Linear lm_head;
    public LayerNorm ln_f;

    public GPT2StyleModel(int n_embd, int n_head, int n_layer, int block_size, int vocab_size, boolean bias) { this(n_embd, n_head, n_layer, block_size, vocab_size, bias, DType.DEFAULT); }
    public GPT2StyleModel(int n_embd, int n_head, int n_layer, int block_size, int vocab_size, boolean bias, Class<?> dtype) {
        this.block_size = block_size;

        this.wte = new Embedding(vocab_size, n_embd);
        this.wpe = new Embedding(block_size, n_embd);
        this.heads = new ArrayList<>();
        for (int i = 0; i < n_layer; i++)
            this.heads.add(new Block(n_embd, n_head, block_size, bias));
        this.lm_head = new Linear(n_embd, vocab_size, false);
        this.ln_f = new LayerNorm(n_embd);
    }

    @SuppressWarnings("unchecked")
    public <T> Tensor<T> forward(Tensor<T> input) {
        // working with Float internally
        Tensor<Double> idx = input.asDType(Double.class);

        int T = idx.size(1); // seq length

        assert T <= this.block_size: "Sequence too large!";

        // embedded vectors
        // shape: b, t, n_embd
        Tensor<Double> x = this.wte.forward(idx)
                .add(this.wpe.forward(Tensor.range(idx.dtype, 0, T, 1)));

        for (Block block : this.heads)
            x = block.forward(x);

        x = this.ln_f.forward(x);

        Tensor<Double> logits = this.lm_head.forward(x); // b, t, voc_size

        return logits.asDType(input.dtype);
    }
}