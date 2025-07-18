package de.c4vxl.core.nn;

import de.c4vxl.core.nn.module.Module;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

import java.util.Arrays;

/**
 * Acts as a lookup table for converting tokens into a fixed embedding-vector.
 * The vectors will be added to the last dimension (-1). Therefor the new shape of an input tensor: [1, 3] will become [1, 3, embedding_dim]
 */
public class Embedding extends Module {
    public Tensor<?> weight;

    public Embedding(int num_embeddings, int embedding_dim) { this(num_embeddings, embedding_dim, DType.DEFAULT); }
    public Embedding(int num_embeddings, int embedding_dim, DType<?> dtype) {
        this.weight = Tensor.ones(num_embeddings, embedding_dim).asDType(dtype);
    }

    public <T> Tensor<T> forward(Tensor<T> x) {
        Integer[] indices = x.asInt().data;

        Integer[] newShape = Arrays.copyOf(x.shape.dimensions.clone(), x.shape.rank() + 1);
        newShape[newShape.length - 1] = this.weight.size(1);
        Tensor<T> result = Tensor.ones(TensorUtils.padShapeLeft(3, true, newShape)).asDType(x.dtype);

        for (int i = 0; i < indices.length; i++) {
            result = result.set(weight.get(indices[i]).asDType(result.dtype), 0, i, null);
        }

        return result;
    }
}