package de.c4vxl.models.type;

import de.c4vxl.engine.activation.Activation;
import de.c4vxl.engine.data.DType;
import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.data.TensorUtils;
import de.c4vxl.engine.module.Module;

import java.util.Objects;

public abstract class TextGenerationModel extends Module {
    // stream of generation
    @FunctionalInterface
    public interface GenerationStream { void apply(int next_token, int index); }

    // forward method
    public abstract <T extends Number> Tensor<T> forward(Tensor<T> input);

    // generation methods
    public <T extends Number> Tensor<T> generate(Tensor<T> input_ids) { return generate(input_ids, 60); }
    public <T extends Number> Tensor<T> generate(Tensor<T> input_ids, int max_new_tokens) { return generate(input_ids, max_new_tokens, 50, null); }
    public <T extends Number> Tensor<T> generate(Tensor<T> input_ids, int max_new_tokens, int block_size, Integer eos_token_id) { return generate(input_ids, max_new_tokens, null, block_size, eos_token_id); }
    public <T extends Number> Tensor<T> generate(Tensor<T> input_ids, int max_new_tokens, GenerationStream stream, int block_size) { return generate(input_ids, max_new_tokens, stream, block_size, null); }
    public <T extends Number> Tensor<T> generate(Tensor<T> input_ids, int max_new_tokens, GenerationStream stream, int block_size, Integer eos_token_id) {
        if (input_ids.is1d()) input_ids = input_ids.unsqueeze(0);
        if (!input_ids.is2d()) throw new IllegalArgumentException("Invalid input_ids shape!");

        for (int i = 0; i < max_new_tokens; i++) {
            // calculate next logits
            Tensor<T> logits = this.forward(
                    TensorUtils.cut_block_size(input_ids, block_size) // cut input ids
            );
            logits = TensorUtils.slice(logits, new int[]{logits.size(0) - 1});
            logits = Activation.Softmax1d(logits.asDType(DType.DOUBLE)).asDType(logits.dtype);

            Integer next_token = TensorUtils.multinomial(logits, 1).item(DType.INTEGER);

            // stream
            if (stream != null)
                stream.apply(next_token, i);

            // stop on eos
            if (Objects.equals(next_token, eos_token_id))
                break;

            // concatenate
            input_ids = input_ids.reshapeUnsafe(1, input_ids.size + 1);
            input_ids.data[input_ids.size-1] = input_ids.valueOf(next_token);
        }

        return input_ids;
    }
}
