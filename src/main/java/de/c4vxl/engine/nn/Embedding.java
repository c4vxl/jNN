package de.c4vxl.engine.nn;

import de.c4vxl.engine.data.DType;
import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.module.Module;

@SuppressWarnings({"rawtypes", "unchecked"})
public class Embedding extends Module {
    public Tensor weight;

    public Embedding(int vocab_size, int embedding_dim) {
        this.weight = Tensor.ones(DType.DEFAULT, vocab_size, embedding_dim);
    }

    @SuppressWarnings("DataFlowIssue")
    public <T> Tensor<T> forward(Tensor<T> tokenIndices) {
        int batchSize = tokenIndices.size(0);
        int seqLength = tokenIndices.size(1);
        int embeddingDim = this.weight.size(1);

        Tensor<T> result = tokenIndices.clone()                     // clone input
                .reshapeUnsafe(batchSize, seqLength, embeddingDim)  // reshape to (b, t, n_embd)
                .fill(tokenIndices.valueOf(0));                     // fill with 0

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLength; t++) {
                int tokenIndex = DType.valueOf(Integer.class, tokenIndices.item(b, t));

                for (int e = 0; e < embeddingDim; e++) {
                    result.set(DType.valueOf(tokenIndices.dtype, this.weight.item(tokenIndex, e)), b, t, e);
                }
            }
        }

        return result;
    }
}