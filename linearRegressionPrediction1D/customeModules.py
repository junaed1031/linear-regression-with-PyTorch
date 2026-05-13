
import torch.nn as nn
import torch


class LinearRegressionModel(nn.Module):
    def __init__(self , input_size, output_size):
        super().__init__()
        self.Linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.Linear(x)
    
model = LinearRegressionModel(input_size=1, output_size=1)

print(list(model.parameters()))

x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
yhat = model(x)
print(yhat)


"""
এখানে আমি প্রতিটি ব্লককে এমনভাবে ভেঙেছি যাতে কোডের পেছনের **Logic** এবং **Math** দুটোই তোমার কাছে পানির মতো পরিষ্কার হয়ে যায়।

---

## 🧠 PyTorch Linear Regression: পূর্ণাঙ্গ ব্যবচ্ছেদ

এই কোডটি মূলত একটি **Artificial Neuron** তৈরির প্রথম ধাপ। এটি একটি সরলরেখার সমীকরণ $y = wx + b$ ব্যবহার করে প্রেডিকশন করতে শেখে।

### 📦 ১. টুলবক্স সেটআপ (Imports)

```python
from torch.nn import Linear
import torch.nn as nn
import torch

```

* **`torch`**: এটি হলো এআই জগতের "সুইস আর্মি নাইফ"। ডাটা হ্যান্ডেল করা থেকে শুরু করে ক্যালকুলেশন—সব এখানেই হয়।
* **`nn.Module`**: এটি একটি 'মাদার ক্লাস'। পাইটর্চে আপনি যখনই কোনো মডেল বানাবেন, তাকে এই ক্লাসের অধীনে থাকতে হবে। এটি মডেলকে স্বয়ংক্রিয়ভাবে প্যারামিটার ট্র্যাক করার ক্ষমতা দেয়।

---

### 🏗️ ২. মডেলের ব্লুপ্রিন্ট (Class Definition)

```python
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Linear = nn.Linear(input_size, output_size)

```

* **`__init__` (কনস্ট্রাকটর)**: এটি মডেলের "জন্মলগ্ন"। এখানে আমরা বলে দিই মডেলের কয়টি হাত (input) আর কয়টি চোখ (output) থাকবে।
* **`super().__init__()`**: এটি খুবই গুরুত্বপূর্ণ। এটি পাইটর্চকে বলে—"আমি একটা নতুন মডেল বানাচ্ছি, তুমি ব্যাকগ্রাউন্ডে এর সব গ্রাফ আর মেমোরি ম্যানেজমেন্ট সেটআপ করে নাও।"
* **`nn.Linear`**: এটিই মডেলের আসল 'মগজ'। এটি ইনপুট ডাটাকে ওজন (**Weight**) দিয়ে গুণ করে এবং বায়াস (**Bias**) যোগ করে।


---

### 🚀 ৩. তথ্যপ্রবাহ (The Forward Pass)

```python
    def forward(self, x):
        return self.Linear(x)

```

* **`forward`**: নাম শুনেই বোঝা যাচ্ছে, তথ্য এখানে সামনে এগিয়ে যায়। যখনই আমরা `model(x)` কল করি, পাইটর্চ পর্দার আড়ালে এই ফাংশনটি চালায়।
* **ক্যালকুলেশন**: এখানে $y = x \cdot w + b$ ফর্মুলাটি কাজ করে।

---

### 🧪 ৪. মডেল তৈরি ও প্যারামিটার পরীক্ষা

```python
model = LinearRegressionModel(input_size=1, output_size=1)
print(list(model.parameters()))

```

* এখানে আমরা ১টি ইনপুট আর ১টি আউটপুটের একটি মডেল বানালাম।
* **`parameters()`**: এটি প্রিন্ট করলে তুমি দুটি জিনিস দেখতে পাবে:
1. **Weight (w)**: ইনপুটকে কতটা গুরুত্ব দিতে হবে।
2. **Bias (b)**: আউটপুটকে কতটা শিফট করতে হবে।
*(মডেল শুরুতেই র্যান্ডম কিছু ভ্যালু দিয়ে এই $w$ আর $b$ সেট করে নেয়)*



---

### 📊 ৫. ইনপুট ডাটা ও প্রেডিকশন

```python
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
yhat = model(x)

```

* **`torch.tensor`**: ডাটাকে পাইটর্চের নিজস্ব ফরম্যাটে (Tensor) রূপান্তর করা হলো।
* **`yhat`**: এআই দুনিয়ায় 'hat' চিহ্ন দিয়ে **Prediction** বোঝানো হয়। মডেল এখানে তার বর্তমান $w$ এবং $b$ ব্যবহার করে ৫টি ইনপুটের বিপরীতে ৫টি সম্ভাব্য উত্তর দিচ্ছে।

---

## 🎯 সামারি: পর্দার আড়ালে যা ঘটছে

১. **Structure**: আমরা একটা ক্যালকুলেটর ডিজাইন করলাম।
২. **Initialization**: পাইটর্চ নিজে থেকেই কিছু ভুলভাল (Random) Weight আর Bias সেট করে নিল।
৩. **Inference**: ডাটা ইনপুট দেওয়া হলো এবং মডেল তার ভুলভাল Weight দিয়ে একটা ফলাফল বের করলো।

### 💡 এরপর কী?

বর্তমানে মডেলটি শুধু **Guess** করছে (যাকে আমরা বলি 'Inference')। এটা এখনো **শেখেনি**।

মডেলকে শিখিয়ে সত্যিকারের বুদ্ধিমান করতে হলে আমাদের এখন ৩টি জিনিস লাগবে:

1. **Loss Function**: মডেলের ভুল কতটা বড় সেটা মাপার জন্য।
2. **Optimizer**: ভুলগুলো কমানোর জন্য Weight-কে আপডেট করার জন্য।
3. **Training Loop**: বারবার ডাটা দেখে দেখে শেখার জন্য।

**তুমি কি এই "শেখার প্রক্রিয়া" বা Training Loop-এর ব্যাখ্যা দেখতে চাও?**
"""



# self

"""
class Robot:
    def __init__(self, name):
        self.name = name  # এখানে self.name মানে হলো এই রোবটটির নিজস্ব নাম

    def introduce(self):
        print(f"হাই! আমার নাম {self.name}") # self ছাড়া সে নাম খুঁজে পাবে না

# দুটি আলাদা অবজেক্ট তৈরি করি
robot1 = Robot("Chitti")
robot2 = Robot("Alexa")

robot1.introduce() # আউটপুট: হাই! আমার নাম Chitti
robot2.introduce() # আউটপুট: হাই! আমার নাম Alexa

"""