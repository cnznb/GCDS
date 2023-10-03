import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

public class RefactoringValidator {
    public static boolean validate(ExtractClassCandidate candidate) {
        if (candidate.getEntities().size() < 2) {
            return false;
        }

        if (candidate.getMethods().isEmpty()) {
            return false;
        }

        if (containsAbstractMethods(candidate)) {
            return false;
        }

        if (containsNonPrivateFields(candidate)) {
            return false;
        }

        if (overridesSuperClassMethods(candidate)) {
            return false;
        }

        if (containsSuperMethodInvocations(candidate)) {
            return false;
        }

        if (containsSynchronizedMethods(candidate)) {
            return false;
        }

        return true;
    }

    private static boolean containsAbstractMethods(ExtractClassCandidate candidate) {
        for (Method method : candidate.getMethods()) {
            if (Modifier.isAbstract(method.getModifiers())) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsNonPrivateFields(ExtractClassCandidate candidate) {
        for (Field field : candidate.getFields()) {
            int modifiers = field.getModifiers();
            if (Modifier.isPublic(modifiers) || Modifier.isProtected(modifiers)) {
                return true;
            }
        }
        return false;
    }

    private static boolean overridesSuperClassMethods(ExtractClassCandidate candidate) {
        for (Method method : candidate.getMethods()) {
            Class<?> superClass = candidate.getOriginalClass().getSuperclass();
            if (superClass != null && isMethodOverridden(method, superClass)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isMethodOverridden(Method method, Class<?> superClass) {
        try {
            superClass.getDeclaredMethod(method.getName(), method.getParameterTypes());
            return true;
        } catch (NoSuchMethodException e) {
            return false;
        }
    }

    private static boolean isMethodOverridden(Method superMethod, Method method) {
        if (!superMethod.getName().equals(method.getName())) {
            return false;  // Method name doesn't match
        }

        Class<?>[] superParamTypes = superMethod.getParameterTypes();
        Class<?>[] paramTypes = method.getParameterTypes();
        if (superParamTypes.length != paramTypes.length) {
            return false;  // Parameter count doesn't match
        }

        for (int i = 0; i < superParamTypes.length; i++) {
            if (!superParamTypes[i].equals(paramTypes[i])) {
                return false;  // Parameter types don't match
            }
        }

        return true;
    }

    private static boolean containsSuperMethodInvocations(ExtractClassCandidate candidate) {
        for (Method method : candidate.getMethods()) {
            if (containsSuperMethodInvocation(method)) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsSuperMethodInvocation(Method method) {
        Class<?> superClass = method.getDeclaringClass().getSuperclass();
        if (superClass == null) {
            return false;  // No super class, so no super method invocation
        }

        Method[] superMethods = superClass.getDeclaredMethods();
        for (Method superMethod : superMethods) {
            if (isMethodOverridden(superMethod, method)) {
                return true;
            }
        }

        return false;
    }

    private static boolean containsSynchronizedMethods(ExtractClassCandidate candidate) {
        for (Method method : candidate.getMethods()) {
            if (Modifier.isSynchronized(method.getModifiers())) {
                return true;
            }
        }
        return false;
    }
}
