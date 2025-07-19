package de.c4vxl.core.optim;

import de.c4vxl.core.optim.type.AbstractOptimizer;
import de.c4vxl.core.tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This is the implementation of an Adam optimizer
 * @see de.c4vxl.core.optim.type.Optimizer
 * @see de.c4vxl.core.optim.AdamWOptimizer
 */
public class AdamOptimizer extends AbstractOptimizer {
    private final double beta1, beta2, epsilon, weight_decay;
    private int timestep;
    private final Map<Tensor<?>, Tensor<?>> m, v;

    public AdamOptimizer(List<Tensor<?>> parameters, double learningRate) {
        this(parameters, learningRate, 0.9, 0.999, 1e-8, 0.01);
    }

    public AdamOptimizer(List<Tensor<?>> parameters, double learningRate, double beta1, double beta2, double epsilon, double weight_decay) {
        super(parameters, learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weight_decay = weight_decay;
        this.timestep = 0;
        this.m = new HashMap<>();
        this.v = new HashMap<>();
    }

    @Override
    public void step() {
        timestep++;

        for (Tensor<?> parameter : this.parameters)
            this.handle(parameter);
    }

    @SuppressWarnings("unchecked")
    private <T> void handle(Tensor<T> parameter) {
        // Initialize moments if necessary
        if (!m.containsKey(parameter)) {
            m.put(parameter, Tensor.zeros(parameter.shape.dimensions).asDType(parameter.dtype));
            v.put(parameter, Tensor.zeros(parameter.shape.dimensions).asDType(parameter.dtype));
        }

        Tensor<T> m_t = (Tensor<T>) m.get(parameter);
        Tensor<T> v_t = (Tensor<T>) v.get(parameter);

        T beta1 = parameter.dtype.parse(this.beta1);
        T beta2 = parameter.dtype.parse(this.beta2);
        T oneMinusBeta1 = parameter.dtype.parse(1 - this.beta1);
        T oneMinusBeta2 = parameter.dtype.parse(1 - this.beta2);

        // m_t = beta1 * m_t + (1 - beta1) * grad
        Tensor<T> m_u = m_t.mul(beta1).add(parameter.grad.mul(oneMinusBeta1));
        m.put(parameter, m_u);

        // v_t = beta2 * v + (1 - beta2) * grad^2
        Tensor<T> gradSquared = parameter.grad.pow(2.);
        Tensor<T> v_u = v_t.mul(beta2).add(gradSquared.mul(oneMinusBeta2));
        v.put(parameter, v_u);

        // Bias correction
        T bc1 = parameter.dtype.parse(1.0 - Math.pow(this.beta1, timestep));
        Tensor<T> m_hat = m_u.div(bc1);
        T bc2 = parameter.dtype.parse(1.0 - Math.pow(this.beta2, timestep));
        Tensor<T> v_hat = v_u.div(bc2);

        // update = learningRate * m_hat / (sqrt(v_hat) + epsilon)
        Tensor<T> denom = v_hat.sqrt().add(parameter.dtype.parse(this.epsilon));
        Tensor<T> step = m_hat.div(denom).mul(parameter.dtype.parse(this.learningRate));

        // Apply weight decay
        Tensor<T> decayTerm = parameter.mul(parameter.dtype.parse(this.learningRate * this.weight_decay));

        Tensor<T> updated = parameter.sub(step).sub(decayTerm);

        // Update parameter
        // Detach to prevent gradients from being reused in next training step
        parameter.update(updated.detach(true));
    }
}