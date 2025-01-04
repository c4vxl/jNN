package de.c4vxl.engine.module;

import com.thoughtworks.xstream.XStream;
import com.thoughtworks.xstream.io.json.JettisonMappedXmlDriver;
import com.thoughtworks.xstream.io.json.JsonHierarchicalStreamDriver;
import com.thoughtworks.xstream.security.NoTypePermission;
import com.thoughtworks.xstream.security.TypePermission;
import de.c4vxl.engine.utils.SerializationUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class Module {
    /**
     * Captures important parameters in a map
     */
    public Map<String, Object> state() {
        Map<String, Object> state = new LinkedHashMap<>();
        SerializationUtils.generateStateRecursively(this, state, "");
        return state;
    }

    /**
     * Load the modules parameters from a state
     * @param state The state to load from
     */
    public Module load_state(Map<String, Object> state) {
        SerializationUtils.loadStateRecursively(this, null, null, state, "");

        return this;
    }

    /**
     * Convert this module into it's json representation
     */
    public String asJson() {
        XStream xStream = new XStream(new JettisonMappedXmlDriver());
        xStream.setMode(XStream.NO_REFERENCES);
        return xStream.toXML(this);
    }

    /**
     * Load this modules state from a json string
     * @param json The json
     */
    @SuppressWarnings("unchecked")
    public static <T extends Module> T fromJson(String json) {
        XStream xStream = new XStream(new JettisonMappedXmlDriver());
        xStream.setMode(XStream.NO_REFERENCES);
        xStream.allowTypesByWildcard(new String[] {
                "de.c4vxl.**",
                "java.lang.**",
                "java.util.**"
        });
        return (T) xStream.fromXML(json);
    }

    /**
     * Export this module into a file
     * @param path The path to the file
     */
    @SuppressWarnings("ResultOfMethodCallIgnored")
    public Module export(String path) {
        try {
            // create file
            File file = new File(path);
            if (!file.exists()) {
                if (file.getParentFile() != null)
                    file.getParentFile().mkdirs();

                file.createNewFile();
            }

            file.setWritable(true);

            // write json
            Files.writeString(file.toPath(), this.asJson());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return this;
    }

    /**
     * Load this module from a file
     * @param path The path to the file
     */
    public static <T extends Module> T load(String path) {
        File file = new File(path);

        if (!file.exists()) {
            System.err.println("Trying to load module from non-existing file. Skipping...");
            return null;
        }

        // load from file
        try {
            return Module.fromJson(Files.readString(file.toPath()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Try to find and invoke a forward function
     * @param args The arguments to pass to the forward function
     */
    public Object _forward(Object... args) {
        try {
            return this.getClass()
                    .getMethod("forward", Arrays.stream(args).map(Object::getClass).toArray(Class[]::new))
                    .invoke(this, args);
        } catch (Exception e) {
            System.err.println("No method for forwarding found!");
            return null;
        }
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + this.state();
    }
}