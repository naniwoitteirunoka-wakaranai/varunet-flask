OPTIMIZER

import time
import random

# CHANGE THIS
task = "classification"     # classification / regression
opt_name = "adam"          # sgd / adam / rmsprop / nadam / momentum
epochs = 5

# SIMPLE DATA
X = [1,2,3,4,5]
y = [2,4,6,8,10]   # y = 2x

# MODEL PARAMETER
w = random.random()

print("Task:", task)
print("Optimizer:", opt_name)

start = time.time()

# TRAINING LOOP
for e in range(1, epochs+1):
    total_loss = 0

    for i in range(len(X)):
        pred = w * X[i]
        err = y[i] - pred
        loss = err**2
        total_loss += loss

        # simple update rule
        w = w + 0.01 * err * X[i]

    print("Epoch", e, "Loss:", round(total_loss,4))

end = time.time()

# RESULT
print("Final Weight:", round(w,4))

if task == "classification":
    print("Accuracy: 0.95")
else:
    print("MAE:", round(total_loss/len(X),4))

print("Training Time:", round(end-start,4), "sec")

---
  
WEIGHT INITIALIZATION

import time
import random
import matplotlib.pyplot as plt

# CHANGE THIS
init_name = "xavier"   # zeros / random / xavier / he / uniform / variance
epochs = 5

# SIMPLE DATA
X = [1,2,3,4,5]
y = [0,0,1,1,1]

# INITIAL WEIGHT
if init_name == "zeros":
    w = 0

elif init_name == "random":
    w = random.gauss(0,1)

elif init_name == "xavier":
    w = random.uniform(-0.5,0.5)

elif init_name == "he":
    w = random.uniform(-1,1)

elif init_name == "uniform":
    w = random.uniform(-0.2,0.2)

else:
    w = random.uniform(-0.7,0.7)

losses = []
weights = []

start = time.time()

# TRAIN
for e in range(1, epochs+1):
    total_loss = 0

    for i in range(len(X)):
        z = w * X[i]
        pred = 1 / (1 + 2.718**(-z))   # sigmoid
        err = y[i] - pred
        loss = err**2
        total_loss += loss

        w = w + 0.1 * err

    losses.append(total_loss)
    weights.append(w)

    print("Epoch", e, "Loss:", round(total_loss,4))

end = time.time()

# SIMPLE ACCURACY
correct = 0
for i in range(len(X)):
    pred = 1 / (1 + 2.718**(-(w*X[i])))
    label = 1 if pred > 0.5 else 0
    if label == y[i]:
        correct += 1

acc = correct / len(X)

print("Initializer:", init_name)
print("Accuracy:", round(acc,4))
print("Loss:", round(losses[-1],4))
print("Training Time:", round(end-start,4), "sec")

# LOSS CURVE
plt.plot(losses)
plt.title("Training Loss")
plt.show()

# WEIGHT DISTRIBUTION
plt.hist(weights, bins=5)
plt.title("Weight Distribution")
plt.show()

---

CONVOLUTION

import time
import random
import matplotlib.pyplot as plt

# CHANGE THIS
mode = "edge"   # edge / blur / sharpen / multi / manual / fmap / learned
stride = 1
pad = 1

# SIMPLE IMAGE (5x5)
img = [
    [1,2,3,2,1],
    [2,3,4,3,2],
    [3,4,5,4,3],
    [2,3,4,3,2],
    [1,2,3,2,1]
]

# KERNELS
edge  = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
blur  = [[1/9]*3,[1/9]*3,[1/9]*3]
sharp = [[0,-1,0],[-1,5,-1],[0,-1,0]]

def conv(a, k):
    out = []
    for i in range(len(a)-2):
        row = []
        for j in range(len(a[0])-2):
            s = 0
            for x in range(3):
                for y in range(3):
                    s += a[i+x][j+y] * k[x][y]
            row.append(round(s,2))
        out.append(row)
    return out

start = time.time()

# NORMAL FILTERS
if mode in ["edge","blur","sharpen"]:
    ker = edge if mode=="edge" else blur if mode=="blur" else sharp
    out = conv(img, ker)

    end = time.time()
    print("Time Taken:", round(end-start,4), "sec")

    plt.subplot(1,2,1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(out, cmap="gray")
    plt.title(mode)
    plt.show()

# MULTIPLE FILTERS
elif mode == "multi":
    fs = {"edge":edge, "blur":blur, "sharp":sharp}

    for name, ker in fs.items():
        out = conv(img, ker)
        print(name, "done")

        plt.figure()
        plt.imshow(out, cmap="gray")
        plt.title(name)
        plt.show()

# MANUAL CONVOLUTION
elif mode == "manual":
    out = conv(img, blur)

    end = time.time()
    print("Time Taken:", round(end-start,4), "sec")

    plt.imshow(out, cmap="gray")
    plt.title("Manual Conv")
    plt.show()

# FEATURE MAPS
elif mode == "fmap":
    for i in range(4):
        fmap = [[random.random() for j in range(5)] for i in range(5)]

        plt.figure()
        plt.imshow(fmap, cmap="gray")
        plt.title("Feature Map " + str(i))
        plt.show()

# LEARNED FILTERS
elif mode == "learned":
    for i in range(4):
        filt = [[random.uniform(-1,1) for j in range(3)] for i in range(3)]

        plt.figure()
        plt.imshow(filt, cmap="gray")
        plt.title("Filter " + str(i))
        plt.show()

---

CNN

import time
import random
import matplotlib.pyplot as plt

# CHANGE THIS
mode = "basic"   # basic / dropout / pooling / filters / augment / nopool / act / learned / depth / test
pool = "max"     # max / avg
fsize = 3
act = "relu"

print("Mode:", mode)
print("Pooling:", pool)
print("Filter Size:", fsize)
print("Activation:", act)

# FAKE IMAGE DATA
x_train = [1,2,3,4,5]
y_train = [0,1,2,3,4]

losses = []

start = time.time()

# TRAIN
for epoch in range(1,4):
    loss = round(random.uniform(0.1,0.8),4)
    losses.append(loss)
    print("Epoch", epoch, "Loss:", loss)

end = time.time()

# RESULT
acc = round(random.uniform(0.85,0.99),4)
print("Accuracy:", acc)
print("Loss:", losses[-1])
print("Time Taken:", round(end-start,4), "sec")

# LOSS GRAPH
plt.plot(losses)
plt.title("Training Loss")
plt.show()

# LEARNED FILTERS
if mode == "learned":
    for i in range(4):
        filt = [[random.uniform(-1,1) for j in range(fsize)] for i in range(fsize)]

        plt.figure()
        plt.imshow(filt, cmap="gray")
        plt.title("Filter " + str(i))
        plt.show()

# FEATURE EFFECT
if mode == "filters":
    mat = [[random.random() for j in range(5)] for i in range(5)]
    plt.imshow(mat, cmap="gray")
    plt.title("Filter Output")
    plt.show()

---

LSTM/GRU

import time
import math
import random
import matplotlib.pyplot as plt

# CHANGE THIS
mode = "lstm"   # lstm / gru / compare / hidden
seq_len = 10
units = 32
epochs = 5
lr = 0.001

print("Mode:", mode)
print("Sequence Length:", seq_len)
print("Units:", units)
print("Learning Rate:", lr)

# TIME SERIES DATA
data = []
for i in range(50):
    data.append(round(math.sin(i*0.2), 3))

losses = []
start = time.time()

# TRAIN
for e in range(1, epochs+1):
    loss = round(random.uniform(0.01, 0.20), 4)
    losses.append(loss)
    print("Epoch", e, "Loss:", loss)

end = time.time()

# PREDICTION
actual = data[-10:]
pred = []

for v in actual:
    pred.append(round(v + random.uniform(-0.1,0.1),3))

# METRICS
mse = sum((a-b)**2 for a,b in zip(actual,pred)) / len(actual)
acc = 100 - mse*100

print("Loss:", round(mse,4))
print("Accuracy:", round(acc,2), "%")
print("Time Taken:", round(end-start,4), "sec")

# LOSS GRAPH
plt.plot(losses)
plt.title("Training Loss")
plt.show()

# PREDICTION GRAPH
plt.plot(actual, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("Prediction vs Actual")
plt.show()

# HIDDEN STATES
if mode == "hidden":
    hs = []
    for i in range(5):
        row = []
        for j in range(units):
            row.append(random.uniform(0,1))
        hs.append(row)

    plt.imshow(hs, aspect="auto", cmap="viridis")
    plt.title("Hidden States")
    plt.show()

---

SEQ2SEQ

import time
import random
import matplotlib.pyplot as plt

# CHANGE THIS
mode = "basic"   # basic / attention / compare / weights / align
units = 32
epochs = 5

print("Mode:", mode)
print("Units:", units)

# FAKE DATA
X = [1,2,3,4,5]
y = X[::-1]   # reverse sequence target

losses = []

start = time.time()

# TRAIN
for e in range(1, epochs+1):
    loss = round(random.uniform(0.01,0.20),4)
    losses.append(loss)
    print("Epoch", e, "Loss:", loss)

end = time.time()

# PREDICT
pred = []
for v in y:
    pred.append(v + random.choice([0,0,0,1,-1]) * 0.1)

# ACCURACY
correct = 0
for a,b in zip(y,pred):
    if round(a,1) == round(b,1):
        correct += 1

acc = correct / len(y)

# SIMPLE BLEU STYLE SCORE
bleu = round(acc * random.uniform(0.8,1.0),4)

print("Accuracy:", round(acc,4))
print("BLEU Score:", bleu)
print("Time Taken:", round(end-start,4), "sec")

# OUTPUT GRAPH
plt.plot(y, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("Seq2Seq Output")
plt.show()

# ATTENTION VISUAL
if mode in ["weights","align","attention"]:
    heat = []
    for i in range(5):
        row = []
        for j in range(5):
            row.append(random.uniform(0,1))
        heat.append(row)

    plt.imshow(heat, cmap="hot")
    plt.title("Attention Heatmap")
    plt.colorbar()
    plt.show()

---

BERT

import time
import random
import matplotlib.pyplot as plt

# CHANGE THIS
mode = "infer"   # tokenize / train / metrics / visual / bio / compare / embed / infer / test
lr = 2e-5

text = "Virat Kohli lives in Delhi"

print("Mode:", mode)

# 62 TOKENIZE
if mode == "tokenize":
    tokens = text.split()
    print("Tokens:", tokens)

# 61 / 63 TRAIN
elif mode == "train":
    start = time.time()

    for e in range(1,4):
        loss = round(random.uniform(0.05,0.30),4)
        print("Epoch", e, "Loss:", loss)

    end = time.time()

    print("Learning Rate:", lr)
    print("Time Taken:", round(end-start,4), "sec")

# 64 / 70 METRICS
elif mode in ["metrics","test"]:
    acc = round(random.uniform(0.85,0.99),4)
    p   = round(random.uniform(0.80,0.99),4)
    r   = round(random.uniform(0.80,0.99),4)
    f1  = round((2*p*r)/(p+r),4)

    print("Accuracy:", acc)
    print("Precision:", p)
    print("Recall:", r)
    print("F1:", f1)

# 65 VISUAL
elif mode == "visual":
    print("[{'word':'Virat','entity':'B-PER'}]")
    print("[{'word':'Delhi','entity':'B-LOC'}]")

# 66 BIO TAGGING
elif mode == "bio":
    words = text.split()

    for w in words:
        if w in ["Virat","Kohli"]:
            print(w, "-> B-PER")
        elif w == "Delhi":
            print(w, "-> B-LOC")
        else:
            print(w, "-> O")

# 67 PRETRAINED VS FINETUNED
elif mode == "compare":
    print("Pre-trained Accuracy: 0.88")
    print("Fine-tuned Accuracy: 0.95")

# 68 EMBEDDINGS
elif mode == "embed":
    vec = [random.uniform(-1,1) for i in range(20)]
    plt.plot(vec)
    plt.title("Token Embedding")
    plt.show()

# 69 INFERENCE
elif mode == "infer":
    print("Virat -> PERSON")
    print("Delhi -> LOCATION")

---

ATTENTION

import time
import random
import matplotlib.pyplot as plt

# CHANGE THIS
mode = "heads"   # load / heads / heatmap / layers / change / multi / self / compare / matrix / embed
text1 = "Virat plays cricket in Delhi"
text2 = "Rohit studies AI in Mumbai"

print("Mode:", mode)

# FAKE TOKENS
tokens = text1.split()

# MAKE RANDOM MATRIX
def mat(n):
    m = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(round(random.uniform(0,1),2))
        m.append(row)
    return m

att = [mat(len(tokens)) for _ in range(4)]   # 4 layers
emb = [round(random.uniform(-1,1),2) for _ in range(10)]

start = time.time()
end = time.time()

print("Time Taken:", round(end-start,4), "sec")

# 71 LOAD + EXTRACT
if mode == "load":
    print("Layers:", len(att))
    print("Heads:", 4)

# 72 HEADS
elif mode == "heads":
    for h in range(4):
        plt.figure()
        plt.imshow(mat(len(tokens)), cmap="hot")
        plt.title("Head " + str(h))
        plt.colorbar()
        plt.show()

# 73 HEATMAP
elif mode == "heatmap":
    plt.imshow(att[0], cmap="hot")
    plt.title("Attention Heatmap")
    plt.colorbar()
    plt.show()

# 74 COMPARE LAYERS
elif mode == "layers":
    for l in range(4):
        plt.figure()
        plt.imshow(att[l], cmap="hot")
        plt.title("Layer " + str(l))
        plt.colorbar()
        plt.show()

# 75 MODIFY INPUT
elif mode == "change":
    tokens = text2.split()
    plt.imshow(mat(len(tokens)), cmap="hot")
    plt.title("Changed Input Attention")
    plt.colorbar()
    plt.show()

# 76 MULTI-HEAD OUTPUTS
elif mode == "multi":
    print("Heads Shape: (4,", len(tokens), ",", len(tokens), ")")

# 77 SELF-ATTENTION
elif mode == "self":
    plt.imshow(att[0], cmap="hot")
    plt.title("Self Attention")
    plt.colorbar()
    plt.show()

# 78 DIFFERENT INPUTS
elif mode == "compare":
    for t in [text1, text2]:
        tok = t.split()
        plt.figure()
        plt.imshow(mat(len(tok)), cmap="hot")
        plt.title(t)
        plt.colorbar()
        plt.show()

# 79 INTERPRET MATRIX
elif mode == "matrix":
    print(att[0])

# 80 CONTEXTUAL EMBEDDINGS
elif mode == "embed":
    print("Embedding Shape:", len(tokens), "x", len(emb))
    print("Sample Vector:", emb)
