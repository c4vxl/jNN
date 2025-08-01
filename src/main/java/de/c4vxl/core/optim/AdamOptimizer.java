package de.c4vxl.core.optim;

import de.c4vxl.core.optim.type.AbstractOptimizer;
import de.c4vxl.core.tensor.Tensor;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class AdamOptimizer extends AbstractOptimizer {
    public static int MAX_MEMORY = 1000;

    private final double beta1, beta2, epsilon;
    private int timestep;
    private final Map<Tensor<?>, Tensor<?>> m, v;

    public AdamOptimizer(List<Tensor<?>> parameters, double learningRate) {
        this(parameters, learningRate, 0.9, 0.999, 1e-8);
    }

    public AdamOptimizer(List<Tensor<?>> parameters, double learningRate, double beta1, double beta2, double epsilon) {
        super(parameters, learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.timestep = 0;
        this.m = new HashMap<>();
        this.v = new HashMap<>();
    }

    @Override
    public void step() {
        timestep++;

        for (Tensor<?> parameter : this.parameters) {
            this.handle(parameter);

            // "Forget" older values
            prune(m);
            prune(v);
        }
    }

    private void prune(Map<Tensor<?>, Tensor<?>> map) {
        while (map.size() > MAX_MEMORY) {
            Iterator<Tensor<?>> it = map.keySet().iterator();
            if (it.hasNext()) {
                it.next();
                it.remove();
            } else {
                break;
            }
        }
    }

    @SuppressWarnings("unchecked")
    protected <T> Tensor<T> calculateStep(Tensor<T> parameter) {
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
        return m_hat.div(denom).mul(parameter.dtype.parse(this.learningRate));
    }

    protected <T> void handle(Tensor<T> parameter) {
        Tensor<T> updated = parameter.sub(this.calculateStep(parameter));

        // Update parameter
        // Detach to prevent gradients from being reused in next training step
        parameter.update(updated.detach(true));
    }
}