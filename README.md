<p align="center">
  <a href="https://github.com/c4vxl/jNN/">
    <img src="https://cdn.c4vxl.de/jNN/media/Logo.svg" alt="Logo">
  </a>
</p>

<h2 align="center">jNN - Java Neural Networking</h2>

<p align="center">jNN, more specifically <b>Java Neural Networking</b>, is a <a href="https://pytorch.org/">PyTorch</a> inspired deep learning library implemented <b>entirely</b> from scratch in Java and is designed for building and training neural networks natively within the ecosystem.</p>

<br>

<p align="center">
  <img src="https://img.shields.io/badge/Programming%20Language:-Java-blue" alt="Java Badge" />
  <img src="https://img.shields.io/badge/Type:-Library-red" alt="Java Badge" />
  <img src="https://img.shields.io/badge/Topic:-AI-orange" alt="Java Badge" />
</p>

---

<br>

### Table of contents (TOC)
- [About jNN](#what-is-jnn)
    - [Tensors in jNN](#tensors-in-jnn)
    - [Autograd/Autodiff](#autograd--automatic-differentiation)

- [Where to find What](#where-to-find-what)
- [Installation](#installation)
    - [Compile yourself](#build-jnn-yourself)
- [Versions](#versions)
- [Resources](#resources)
- [License](#license)

<br>

# What is jNN?
**jNN** (short for Java Neural Networking) is a deep learning library, inspired by libraries like [PyTorch](https://pytorch.org/), build entirely from scratch in Java. It provides the tools needed to define, train, and deploy neural networks natively within the Java ecosystem, without relying on external bindings or JNI.

jNN supports essential components for machine learning, including:

- A powerful Tensor library with support for multiple data types and operations.
- An autograd engine that automatically tracks operations and computes gradients for backpropagation.
- Reusable neural network modules, optimizers, training utilities, and more.

It’s designed to bring a clean and expressive machine learning experience to Java and is ideal for education, experimentation, or integrating ML into existing Java-based projects.

<br>

# Tensors in jNN
In **jNN**, a `Tensor` is the core data structure used to represent multi-dimensional arrays. These Tensor-Objects are capable of performing various mathematical operations such as matrix multiplication, broadcasting, or standard operations like `addition`, `subtraction`, etc.
Additionally, Tensors support multiple java-native data types (`DTypes` for short), such as Double, Long, Integer or even truthy types like Boolean and can be converted seamlessly from one to another.

Tensors are implemented to support autograd, meaning they can automatically build out a graph of operations which created them and use it for backpropagation. [[Read more about autograd]](#autograd--automatic-differentiation).

**There are many different ways of constructing new Tensors, but here is a common one:**
```java
// This creates a 4x2 Tensor filled with random values ranging from 0 to 100
Tensor<Double> a = Tensor.random(DType.DOUBLE, 0, 100, 4, 2)
```

*Operations are performed by calling methods on a tensor and passing additional operants as arguments*:
```java
// This adds 5 to b and transposes the result over the last two dimensions
// Result is a 2x4 Tensor
Tensor<Double> b = a.add(5.0).transpose(-1, -2);
```

<br>


# Autograd / Automatic Differentiation
![Preview](http://cdn.c4vxl.de/jNN/media/autograd_preview.svg)

jNN includes a lightweight autograd engine that powers gradient-based learning.

While performing operations on Tensor objects with gradient tracking enabled, jNN builds a computational graph in the background, where every Tensor keeps a reference to the operation and Tensors used to compute it.

During backpropagation, **jNN** traverses this graph in reverse to compute gradients for every operation.

**Example:**
```java
Tensor<Double> x = Tensor.of(2.);     // Tensor holding a 2
Tensor<Double> y = x.mul(x).add(x);   // y = x^2 + x
y.backward();                         // Computes dy/dx
System.out.println(x.grad());         // 5.0 (dy/dx at x=2)
```
```
y = x² + x
-> ∂y/∂x = 2x + 1
-> ∂y/∂x(2) = 5
```

<br>

# Where to find what:
*Here are some direct links to the different sections of the jNN code base:*
| Component | Description |
| --------- | ----------- |
| [**/core/nn**](https://github.com/c4vxl/jNN/tree/main/src/main/java/de/c4vxl/core/nn/) | A collection of reusable modules commonly used in neural networks. |
| [**/core/tensor**](https://github.com/c4vxl/jNN/tree/main/src/main/java/de/c4vxl/core/tensor/) | The core tensor library that powers all mathematical operations, autodiff, and data manipulation. |
| [**/core/optim**](https://github.com/c4vxl/jNN/tree/main/src/main/java/de/c4vxl/core/optim/) | Implementations of optimizers that update model parameters based on gradients computed by the autograd engine. |
| [**/core/utils**](https://github.com/c4vxl/jNN/tree/main/src/main/java/de/c4vxl/core/utils/) | Useful utility functions for all kind of data manipulation, serialization or initialization. |
| [**/models**](https://github.com/c4vxl/jNN/tree/main/src/main/java/de/c4vxl/models/) | Predefined, ready-to-use model architectures. Like MLPs, Transformers or LSTMs |
| [**/train**](https://github.com/c4vxl/jNN/tree/main/src/main/java/de/c4vxl/train/) | A collection of training loops. |
| [**/tokenizers**](https://github.com/c4vxl/jNN/tree/main/src/main/java/de/c4vxl/tokenizers/) | Implementations of tokenizers used for [TextGenerationModels](/src/main/java/de/c4vxl/models/type/TextGenerationModel.java). |

<br>

# Installation
> [!NOTE]
> jNN was build using a JDK of version 21.
> Other versions are experimental only!

<br>

Since **jNN** is meant to be used as a Java library, you can add it to your project by including the jNN Maven repository and specifying the desired version as a dependency.

<details open>
<summary><span style="font-size: 2rem">Gradle</span></summary>

*Here's how to import jNN into your Gradle project:*

**build.gradle.kts**:
```gradle
...

repositories {
    maven("https://mvn.c4vxl.de/jNN/") // Add jNN-repo
}

dependencies {
    implementation("de.c4vxl:jNN:1.0.0") // Import specific version
}

...
```

or

**build.gradle**:
```gradle
...

repositories {
    maven {
        url = uri("https://mvn.c4vxl.de/jNN/") // Add jNN-repo
    }
}

dependencies {
    implementation 'de.c4vxl:jNN:1.0.0' // Import specific version
}

...
```
</details>

<details>
<summary><span style="font-size: 2rem">Maven</span></summary>

*Here's how to import jNN into your Maven project:*

**pom.xml**:
```xml
<project>
  ...
  <repositories>
    <repository>
      <id>jNN</id>
      <url>https://mvn.c4vxl.de/jNN/</url>
    </repository>
  </repositories>

  <dependencies>
    <dependency>
      <groupId>de.c4vxl</groupId>
      <artifactId>jNN</artifactId>
      <version>1.0.0</version>
    </dependency>
  </dependencies>
  ...
</project>
```
</details>

<br>

# Build jNN yourself
To compile **jNN** youself, simply follow these steps:
1. Clone this repository
2. Build it using gradle: `./gradlew clean build`
3. The compiled `.jar` file can be found at /build/libs/

<br>

# Versions
| Version | Description | Commit |
| ----------- | ----------- | -----------  |
| **1.0.0** (_latest_) | First release version of jNN. | [4e8ecd355844f32ba253c8d80e89d75ae6d7aff2](https://github.com/c4vxl/jNN/commit/4e8ecd355844f32ba253c8d80e89d75ae6d7aff2) |

<br>

# Resources
- [Example Project](https://github.com/c4vxl/jNN_playground/)
- [Documentation](https://docs.c4vxl.de/jNN/)

<br>

# License
You are welcome to use and experiment with this library freely for personal, educational, or research purposes.

Note that:
- If you use jNN in your own projects, you must credit the original library / author (c4vxl).
- For any larger-scale, commercial, or production use, or if you plan to redistribute or modify jNN, you are required contact the author first to discuss terms.

This library is provided as-is, without warranties, and is not optimized for high-performance production environments.


---

> [!IMPORTANT]
> This project was built by a single person as a fun and educational endeavor.
>
> Since jNN is written entirely in Java, there are inherent performance limitations compared to libraries built in lower-level languages like C++. While I did my best in trying to implement efficient algorithms, this library may not be suitable for high-performance or in-production environments.
>
> That said, jNN is robust and works very well for smaller-scale models and tasks such as local image recognition.

<br>

A project by [c4vxl](https://info.c4vxl.de/)
