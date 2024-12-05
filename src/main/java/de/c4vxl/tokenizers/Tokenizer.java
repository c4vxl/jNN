package de.c4vxl.tokenizers;

import de.c4vxl.engine.data.Tensor;

import java.util.Arrays;
import java.util.List;

public interface Tokenizer {
    default Tensor<Integer> encode(String input) { return Tensor.of(this._encode(input).toArray(Integer[]::new)); }
    List<Integer> _encode(String input);

    default String decode(Tensor<Integer> input_ids) { return this._decode(Arrays.stream(input_ids.data).toList()); };
    String _decode(List<Integer> input_ids);
}