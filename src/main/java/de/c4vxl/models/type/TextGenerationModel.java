package de.c4vxl.models.type;

import de.c4vxl.core.activation.ActivationFunction;
import de.c4vxl.core.module.Module;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.utils.TensorUtils;

import java.util.Objects;

/**
 * Base model for text generation models.
 * This class implements basic features for generation of a new token sequence
 */
public abstract class TextGenerationModel extends Module {
    @FunctionalInterface
    public interface GenerationStream { void apply(int next_token, int idx); }

    public abstract <T extends Number> Tensor<Double> forward(Tensor<T> input);


    public <T extends Number> Tensor<T> generate(Tensor<T> input_ids, int max_new_tokens, int block_size) {
        return generate(input_ids, 1.0, max_new_tokens, block_size, null, null);
    }
    public <T extends Number> Tensor<T> generate(Tensor<T> input_ids, int max_new_tokens, int block_size, GenerationStream stream) {
        return generate(input_ids, 1.0, max_new_tokens, block_size, null, stream);
    }
    public <T extends Number> Tensor<T> generate(Tensor<T> input_ids, double temperature, int max_new_tokens, int block_size, Integer eos_token_id, GenerationStream stream) {
        if (input_ids.shape.rank() == 1) input_ids = input_ids.unsqueeze(0);
        if (input_ids.shape.rank() != 2)
            throw new IllegalArgumentException("`input_ids` can only be 1d or 2d!");

        for (int i = 0; i < max_new_tokens; i++) {
            // "forget" older tokens by narrowing down the token dim to be at most "block_size" tokens long
            int dimSize = input_ids.size(1);
            if (dimSize > block_size)
                input_ids = TensorUtils.narrow(input_ids, 1, dimSize - block_size, block_size);

            // forward through model
            Tensor<Double> logits = this.forward(input_ids);
            logits = logits.get(null, -1, null); // only take the last time dimension
            if (logits.shape.rank() == 1)
                logits = logits.unsqueeze(0);

            // get next token
            logits = logits.div(temperature);
            logits = ActivationFunction.Softmax(logits, temperature, -1);
            Integer nextToken = TensorUtils.multinomial(logits, 1).squeeze().item(0);

            if (stream != null)
                stream.apply(nextToken, i);

            // stop on eos_token
            if (eos_token_id != null && Objects.equals(nextToken, eos_token_id))
                break;

            // append nextToken
            input_ids = input_ids.reshapeUnsafe(input_ids.size(0), input_ids.size(1) + 1);
            input_ids.data[input_ids.size() - 1] = input_ids.dtype.parse(nextToken);
        }

        return input_ids;
    }
}