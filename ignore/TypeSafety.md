Based on the article you provided and the current state of the Python ecosystem, the **"best modern way"** is not a single tool, but a combination of **Type Annotations** (for development) and **Pydantic** (for runtime safety).

If you are coming from a **TypeScript** background, the transition looks like this:

### 1. The "TypeScript" of Python: Type Annotations + Static Checkers

Just like TS code is checked by the `tsc` compiler before running, Python uses **Type Annotations**. However, Python's interpreter ignores these by default. To get "TypeScript-like" behavior, you must use a static type checker:

- **Mypy** or **Pyright**: These are the industry standards.
- **VS Code (Pylance)**: This uses Pyright under the hood to give you those red squiggly lines when you pass a `str` where an `int` should be.

### 2. The "Interface" Equivalent: Pydantic

In TypeScript, an `interface` ensures an object has a certain shape. In modern Python, **Pydantic** is the gold standard for this.

**Why Pydantic is considered the "Best Modern Way":**

- **Data Parsing/Coercion:** If you pass the string `"5"` to an `int` field, Pydantic automatically converts it to `5`.
- **Runtime Enforcement:** Unlike standard Type Annotations or Data Classes, Pydantic will throw a `ValidationError` the moment you try to create an object with the wrong data types.
- **JSON Integration:** It is built for the modern web (FastAPI, the most popular modern Python web framework, is built entirely on Pydantic).

---

### Comparison at a Glance

| Feature              | Type Annotations       | Data Classes        | Pydantic                |
| -------------------- | ---------------------- | ------------------- | ----------------------- |
| **Primary Use**      | Documentation/IDE help | Simple data storage | API data & Validation   |
| **Runtime Safety**   | ❌ None                | ❌ None             | ✅ Strict enforcement   |
| **Data Conversion**  | ❌ Manual              | ❌ Manual           | ✅ Automatic (Coercion) |
| **Standard Library** | ✅ Yes                 | ✅ Yes              | ❌ External Library     |

---

### Which one should you use?

1. **Use Type Annotations** everywhere. It’s the baseline for modern Python.
2. **Use Data Classes** for internal logic where you trust your data source and want to keep dependencies low (no external libraries).
3. **Use Pydantic** for anything involving "The Outside World" (API calls, User Input, JSON files). It is the closest experience you will get to the safety and productivity of TypeScript.

**Would you like me to show you how to convert a specific TypeScript interface into a Pydantic model?**
