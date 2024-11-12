package de.c4vxl.engine.nn;

import de.c4vxl.engine.data.Module;
import de.c4vxl.engine.data.Tensor;

/**
 * Applies layer normalization to stabilize and improve model training.
 *
 * @apiNote EXPERIMENTAL
 * @author c4vxl
 */
@SuppressWarnings({"unchecked", "rawtypes"})
public class LayerNorm extends Module {
    private Tensor gamma;
    private Tensor beta;
    private final double epsilon;
    private final int[] normalizedShape;

    public LayerNorm(double epsilon, int... normalizedShape) {
        this.normalizedShape = normalizedShape;
        this.epsilon = epsilon;
        this.gamma = Tensor.ones(normalizedShape);
        this.beta = Tensor.zeros(normalizedShape);
    }

    public LayerNorm(int... normalizedShape) {
        this(1e-5, normalizedShape);
    }

    public Tensor forward(Tensor input) {
        int[] shape = input.shape;
        int normalizedAxis = shape.length - normalizedShape.length;

        // mean and variance along the specified axis
        Tensor mean = input.sum(normalizedAxis).div(normalizedShape[0]);
        Tensor variance = input.sub(mean).pow(2).sum(normalizedAxis).div(normalizedShape[0]);

        // normalize the input
        Tensor normalized = input.sub(mean.unsqueeze(normalizedAxis))
                .div(variance.add(epsilon).sqrt().unsqueeze(normalizedAxis));

        // apply gamma and beta
        return normalized.mul(gamma).add(beta);
    }
}