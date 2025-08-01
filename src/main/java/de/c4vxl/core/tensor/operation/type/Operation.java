package de.c4vxl.core.tensor.operation.type;

import de.c4vxl.core.tensor.Tensor;

import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public abstract class Operation<T> {
    protected List<Tensor<T>> inputs;
    protected HashMap<String, Object> cache;

    @SafeVarargs
    public Operation(Tensor<T>... inputs) {
        this.inputs = Arrays.stream(inputs).toList();
        this.cache = new HashMap<>();
    }

    /**
     * Store a value for later access
     * @param key The key to the variable
     * @param value The value
     */
    public void saveForBackward(String key, Object value) {
        this.cache.put(key, value);
    }

    /**
     * Get a stored value
     * @param key The key to the variable
     */
    @SuppressWarnings("unchecked")
    public <R> R getValue(String key) {
        if (!this.cache.containsKey(key))
            return null;

        return (R) this.cache.get(key);
    }

    public abstract Tensor<T> _forward();
    public abstract void _backward(Tensor<T> gradOutput);

    /**
     * Invoke this operation with the parameters passed in the constructor
     */
    public Tensor<T> forward() {
        Tensor<T> result = this._forward();
        result.requires_grad = this.inputs.stream().anyMatch(inp -> inp.requires_grad);
        result.operation = this;
        result.parents = List.of(this.inputs.toArray(Tensor<?>[]::new));
        result.is_leaf = false;
        return result;
    }

    /**
     * Perform a backward pass through this operation
     * @param gradOutput The gradient output from the next node in the graph
     */
    @SuppressWarnings("unchecked")
    public void backward(Tensor<?> gradOutput) {
        this._backward((Tensor<T>) gradOutput);
    }
}