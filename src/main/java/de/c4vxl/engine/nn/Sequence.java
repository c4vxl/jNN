package de.c4vxl.engine.nn;

import de.c4vxl.engine.data.Module;
import de.c4vxl.engine.data.Tensor;

import java.util.List;

/**
 * A class to run multiple Modules in a sequence
 *
 * @author c4vxl
 */
public class Sequence extends Module {
    private final List<Module> modules;

    public Sequence(Module... sequence) {
        this.modules = List.of(sequence);
    }

    public Sequence add(Module module) {
        modules.add(module);
        return this;
    }

    public Sequence add(Module module, int index) {
        modules.add(index, module);
        return this;
    }

    @SuppressWarnings("unchecked")
    public <T> Tensor<T> forward(Tensor<T> x) {
        for (int i = 0; i < modules.size(); i++) {
            try {
                Module module = modules.get(i);
                x = (Tensor<T>) module.getClass().getMethod("forward", Tensor.class).invoke(module, x);
            } catch (Exception e) {
                System.err.println("Sequence " + this.getClass() + " couldn't forward through module " + i + ". Skipping!");
            }
        }

        return x;
    }
}