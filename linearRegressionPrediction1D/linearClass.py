import torch
from torch.nn import Linear

torch.manual_seed(1)

model = Linear(in_features=1, out_features=1)

print(list(model.parameters()))

x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = model(x)
print(yhat)





"""
এই কোডটা একটা খুব ছোট **PyTorch Neural Network model** তৈরি করছে।
ধাপে ধাপে সহজভাবে বুঝাই:

---

# পুরো কোড

```python
import torch
from torch.nn import Linear

torch.manual_seed(1)

model = Linear(in_features=1, out_features=1)

print(list(model.parameters()))

x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = model(x)
print(yhat)
```

---

# ১) torch import করা

```python
import torch
```

এখানে PyTorch লাইব্রেরি import করা হয়েছে।

PyTorch ব্যবহার হয়:

* Deep Learning
* Neural Network
* Tensor calculation
* AI model training

এর জন্য।

---

# ২) Linear layer import করা

```python
from torch.nn import Linear
```

এখানে `Linear` layer import করা হয়েছে।

এটা আসলে Neural Network এর সবচেয়ে basic layer।

এটার equation:

[
y = wx + b
]

যেখানে:

* `w` = weight
* `b` = bias

---

# ৩) Random seed set করা

```python
torch.manual_seed(1)
```

Computer random number generate করে।

Neural Network এর weight শুরুতে random হয়।

`manual_seed(1)` দিলে:

* প্রতিবার একই random value আসবে
* ফলে output repeatable হবে

মানে:
আজ run করলে যা আসবে,
আগামীকালও একই output আসবে।

---

# ৪) Model তৈরি করা

```python
model = Linear(in_features=1, out_features=1)
```

এখানে একটি Linear model তৈরি হয়েছে।

মানে:

* Input feature = 1
* Output feature = 1

এটা basically:

[
y = wx + b
]

এই equation শিখবে।

---

## in_features=1

মানে input এ ১টা value থাকবে।

যেমন:

```python
[1.0]
```

---

## out_features=1

মানে output এ ১টা value দিবে।

---

# Model internally কী বানায়?

PyTorch automatically বানায়:

* Weight (`w`)
* Bias (`b`)

উদাহরণ:

```python
w = 0.5
b = -0.2
```

এগুলো random initial value।

---

# ৫) Parameters print করা

```python
print(list(model.parameters()))
```

এটা model এর:

* weight
* bias

print করে।

উদাহরণ output এরকম হতে পারে:

```python
[Parameter containing:
tensor([[0.5153]], requires_grad=True),
 Parameter containing:
tensor([-0.4414], requires_grad=True)]
```

এখানে:

* `0.5153` = weight
* `-0.4414` = bias

---

# requires_grad=True মানে কী?

এটার মানে:
PyTorch training এর সময় automatically gradient calculate করবে।

এটা Deep Learning training এর জন্য দরকার।

---

# ৬) Input tensor তৈরি

```python
x = torch.tensor([[1.0], [2.0], [3.0]])
```

এখানে input data তৈরি হয়েছে।

এটা দেখতে:

[
\begin{bmatrix}
1.0 \
2.0 \
3.0
\end{bmatrix}
]

---

## কেন double bracket?

কারণ PyTorch চায়:

```python
[number_of_samples, number_of_features]
```

এখানে:

* 3 samples
* প্রতি sample এ 1 feature

তাই shape হবে:

```python
(3,1)
```

---

# ৭) Model দিয়ে prediction করা

```python
yhat = model(x)
```

এখানে model calculation করছে:

[
y = wx + b
]

ধরি:

* w = 0.5153
* b = -0.4414

তাহলে:

---

## প্রথম input

x = 1

[
y = (0.5153)(1) - 0.4414
]

[
y = 0.0739
]

---

## দ্বিতীয় input

x = 2

[
y = (0.5153)(2) - 0.4414
]

[
y = 0.5892
]

---

এভাবেই সব input এর output বের হয়।

---

# ৮) Output print করা

```python
print(yhat)
```

এটা prediction output দেখাবে।

উদাহরণ:

```python
tensor([[0.0739],
        [0.5891],
        [1.1044]], grad_fn=<AddmmBackward0>)
```

---

# grad_fn=<AddmmBackward0> মানে কী?

এটা PyTorch এর Autograd system।

মানে:
PyTorch জানে কীভাবে backward propagation করতে হবে।

Training এর সময় এটা কাজে লাগে।

---

# পুরো flow এক লাইনে

```text
Input x
   ↓
Linear layer (y = wx+b)
   ↓
Output yhat
```

---

# এই কোড আসলে কী করছে?

এটা একটা খুব ছোট neural network তৈরি করছে যা:

[
y = wx + b
]

এই equation ব্যবহার করে input থেকে output বের করছে।

এখনও training হয়নি,
শুধু random weight দিয়ে prediction দিচ্ছে।

---

# Beginner-friendly analogy

ভাবো:

```text
Input → Machine → Output
```

Machine এর ভিতরে:

* weight
* bias

আছে।

PyTorch সেই machine বানিয়ে দিয়েছে।

"""