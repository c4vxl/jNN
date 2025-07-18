package de.c4vxl.core.nn;

import de.c4vxl.core.nn.module.Module;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A Module used for stacking multiple components in sequence.
 */
public class Sequence extends Module {
    public ArrayList<Module> modules = new ArrayList<>();

    public Sequence(Module... modules) { this.append(modules); }

    /**
     * Append one or multiple modules at the end of the sequence
     * @param modules The list of modules
     */
    public Sequence append(Module... modules) {
        this.modules.addAll(Arrays.stream(modules).toList());
        return this;
    }

    /**
     * Append a module in at specific position
     * @param module The module to add
     * @param idx The position for the module
     */
    public Sequence append(Module module, int idx) {
        this.modules.add(idx, module);
        return this;
    }

    /**
     * Remove a module at a given position from the sequence
     * @param idx The position of the module
     */
    public Sequence remove(int idx) {
        this.modules.remove(idx);
        return this;
    }

    /**
     * Remove a module from the sequence
     * @param module The module to remove
     */
    public Sequence remove(Module module) {
        this.modules.remove(module);
        return this;
    }

    /**
     * Forward an input through the sequence
     * @param input The input
     */
    @SuppressWarnings("unchecked")
    public <T> T forward(Object input) {
        for (Module module : this.modules)
            input = module._forward(input);

        return (T) input;
    }
}
