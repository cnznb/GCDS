import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

public class ExtractClassCandidate {
    private Class<?> originalClass;
    private List<Field> fields;
    private List<Method> methods;
    private List<Object> entities;

    public ExtractClassCandidate(Class<?> originalClass) {
        this.originalClass = originalClass;
        this.fields = new ArrayList<>();
        this.methods = new ArrayList<>();
        this.entities = new ArrayList<>();
    }

    public Class<?> getOriginalClass() {
        return originalClass;
    }

    public List<Field> getFields() {
        return fields;
    }

    public List<Method> getMethods() {
        return methods;
    }

    public List<Object> getEntities() {
        return entities;
    }

    public void addField(Field field) {
        fields.add(field);
    }

    public void addMethod(Method method) {
        methods.add(method);
    }

    public void addEntity(Object entity) {
        entities.add(entity);
    }
}
