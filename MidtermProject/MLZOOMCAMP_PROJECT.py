#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().system('pip install numpy pandas matplotlib scikit-learn tensorflow pillow kaggle opencv-python')


# In[18]:


import os
from zipfile import ZipFile
from pathlib import Path

# --- change this only if the path is different ---
ZIP_PATH = r"C:\Users\Sumana Sarkar\Downloads\galaxy-zoo-the-galaxy-challenge.zip"

# Where we'll extract selected inner zips
WORK_DIR = Path.cwd() / "galaxy_data"   # notebook working directory: ./galaxy_data
WORK_DIR.mkdir(exist_ok=True)

print("Notebook working dir:", WORK_DIR)
print("Top-level zip path exists:", Path(ZIP_PATH).exists())

# List top-level zip contents
with ZipFile(ZIP_PATH, 'r') as z:
    names = z.namelist()
print("Top-level zip contains", len(names), "entries; sample:")
for p in names[:50]:
    print(" ", p)


# In[19]:


inner_to_extract = [
    "images_training_rev1.zip",
    "training_solutions_rev1.zip",
    "images_test_rev1.zip"
]

with ZipFile(ZIP_PATH, 'r') as z:
    for inner in inner_to_extract:
        if inner in z.namelist():
            target = WORK_DIR / inner
            if target.exists():
                print(f"{inner} already exists; skipping extract.")
            else:
                print("Extracting", inner, "to", target)
                with z.open(inner) as inner_file, open(target, 'wb') as f:
                    f.write(inner_file.read())
        else:
            print("WARNING: inner zip not found in top-level zip:", inner)


# In[20]:


# destination folders
TRAIN_IMG_DIR = WORK_DIR / "images_training"
TEST_IMG_DIR  = WORK_DIR / "images_test"
LABELS_DIR   = WORK_DIR / "labels"
TRAIN_IMG_DIR.mkdir(exist_ok=True)
TEST_IMG_DIR.mkdir(exist_ok=True)
LABELS_DIR.mkdir(exist_ok=True)

# helper to unzip inner zip to target dir
def unzip_file(zipfile_path, target_dir, show_count=10):
    from zipfile import ZipFile
    zipfile_path = Path(zipfile_path)
    print(f"Unzipping {zipfile_path} → {target_dir}")
    with ZipFile(zipfile_path, 'r') as z:
        z.extractall(target_dir)
    print("Done. files in target (sample):")
    sample = list(Path(target_dir).glob("*"))[:show_count]
    for s in sample:
        print(" ", s.name)

# Extract training images (this is large; sample mode below)
inner_train_zip = WORK_DIR / "images_training_rev1.zip"
if inner_train_zip.exists():
    unzip_file(inner_train_zip, TRAIN_IMG_DIR)
else:
    print("images_training_rev1.zip not found at", inner_train_zip)

# Extract labels CSV
inner_labels_zip = WORK_DIR / "training_solutions_rev1.zip"
if inner_labels_zip.exists():
    # unzip into LABELS_DIR; should contain a CSV
    unzip_file(inner_labels_zip, LABELS_DIR)
else:
    print("training_solutions_rev1.zip not found at", inner_labels_zip)

# optional: test zip
inner_test_zip = WORK_DIR / "images_test_rev1.zip"
if inner_test_zip.exists():
    unzip_file(inner_test_zip, TEST_IMG_DIR)
else:
    print("images_test_rev1.zip not found (test set).")


# In[21]:


import pandas as pd
import glob

# find the CSV inside LABELS_DIR
csv_files = list(LABELS_DIR.glob("*.csv")) + list(LABELS_DIR.glob("*.tsv"))
print("Found label files:", csv_files)
labels_csv = csv_files[0]  # choose first found
print("Using labels file:", labels_csv)

labels = pd.read_csv(labels_csv)
labels.head()


# In[22]:


print("shape:", labels.shape)
cols = labels.columns.tolist()
print("first 20 columns:", cols[:20])
print("last 20 columns:", cols[-20:])
# Detect vote-fraction columns (exclude image id column names commonly 'img_id' or 'IMG_ID' etc.)
vote_cols = [c for c in cols if c.lower() not in ('img_id','image_id','imgid','id','img')]
len(vote_cols), vote_cols[:10]


# In[23]:


# Determine vote-fraction columns (exclude GalaxyID-like columns)
possible_id_names = {'GalaxyID','galaxyid','img_id','IMG_ID','ImageId','id','ID'}
cols = labels.columns.tolist()
vote_cols = [c for c in cols if c not in possible_id_names]
print("Number of vote columns found:", len(vote_cols))
print("First vote columns:", vote_cols[:12])
# create unified image id column
if 'GalaxyID' in labels.columns:
    labels = labels.rename(columns={'GalaxyID':'img_id'})
elif 'img_id' in labels.columns:
    labels = labels.rename(columns={'img_id':'img_id'})
else:
    # fallback: first column assumed to be id
    labels = labels.rename(columns={labels.columns[0]:'img_id'})
print("Using image id column:", 'img_id')
# Top answer (column with largest vote fraction)
labels['top_answer_col'] = labels[vote_cols].idxmax(axis=1)
labels[['img_id','top_answer_col']].head()


# In[24]:


def map_top_to_coarse(colname):
    s = str(colname).lower()
    # Heuristic mapping tuned for Galaxy Zoo ClassX.Y naming:
    # Class1.* typically corresponds to top-level "smooth / features or disk / star/artifact"
    # Class1.1 -> smooth (elliptical), Class1.2 -> features/disk (spiral), Class1.3 -> star/artifact (other)
    if s.startswith('class1.1') or 'class1.1' in s:
        return 'elliptical'
    if s.startswith('class1.2') or 'class1.2' in s:
        return 'spiral'
    if s.startswith('class1.3') or 'class1.3' in s:
        return 'other'
    # fallback lookups for keywords (in case column names contain words)
    if 'smooth' in s or 'elliptical' in s:
        return 'elliptical'
    if 'spiral' in s or 'arm' in s or 'disk' in s or 'features' in s:
        return 'spiral'
    if 'merger' in s or 'odd' in s or 'artifact' in s or 'star' in s:
        return 'other'
    # last fallback
    return 'other'

labels['target'] = labels['top_answer_col'].apply(map_top_to_coarse)
print(labels['target'].value_counts())
labels[['img_id','top_answer_col','target']].head(10)


# In[25]:


# Some CSVs use GalaxyID without leading zeros; images are <GalaxyID>.jpg
# check first few ids, and whether images exist
sample_ids = labels['img_id'].astype(str).head(10).tolist()
print("sample ids:", sample_ids)
for sid in sample_ids:
    p = TRAIN_IMG_DIR / f"{sid}.jpg"
    print(sid, "exists?", p.exists())
# show a sample image if available
for sid in sample_ids[:3]:
    p = TRAIN_IMG_DIR / f"{sid}.jpg"
    if p.exists():
        display(Image.open(p))
    else:
        print("Missing image for id:", sid)


# In[28]:


from pathlib import Path

WORK_DIR = Path(r"C:/Users/Sumana Sarkar/galaxy_data")   # your working folder
print("WORK_DIR:", WORK_DIR, "exists?", WORK_DIR.exists())

jpgs = list(WORK_DIR.rglob("*.jpg"))
print("Total .jpg files found under WORK_DIR:", len(jpgs))
print("Sample (first 40):")
for p in jpgs[:40]:
    print(" ", p)


# In[29]:


from pathlib import Path

WORK_DIR = Path(r"C:/Users/Sumana Sarkar/galaxy_data")
print("WORK_DIR exists?", WORK_DIR.exists())

# find folder with the largest number of jpgs (search depth arbitrary)
best_dir = None
best_count = 0
for d in WORK_DIR.rglob("*"):
    if d.is_dir():
        cnt = sum(1 for _ in d.glob("*.jpg"))
        if cnt > best_count:
            best_count = cnt
            best_dir = d

print("Best candidate folder with most .jpg files:", best_dir)
print("Number of .jpg files there:", best_count)
# list a sample of files
if best_dir:
    sample = list(best_dir.glob("*.jpg"))[:20]
    print("Sample files:", [p.name for p in sample])


# In[30]:


from pathlib import Path
WORK_DIR = Path(r"C:/Users/Sumana Sarkar/galaxy_data")

# 1) find folders with many .jpg files and print counts (helps locate training folder)
candidates = []
for d in WORK_DIR.rglob("*"):
    if d.is_dir():
        cnt = sum(1 for _ in d.glob("*.jpg"))
        if cnt>0:
            candidates.append((d, cnt))
# sort descending by count
candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
print("Top candidate folders with jpg counts (top 6):")
for d,c in candidates[:6]:
    print(f"  {d}  -> {c}")

# 2) auto-select the folder whose jpg count is closest to the labels length (61578)
labels_count = 61578
best = min(candidates, key=lambda x: abs(x[1]-labels_count)) if candidates else (None,0)
print("\nAuto-picked best candidate for TRAIN_IMG_DIR (closest to 61578 images):")
print(best)
TRAIN_IMG_DIR = best[0]
print("Set TRAIN_IMG_DIR =", TRAIN_IMG_DIR)


# In[31]:


import pandas as pd
LABELS_CSV = WORK_DIR / "labels" / "training_solutions_rev1.csv"
labels = pd.read_csv(LABELS_CSV)
# unify id column name
if 'GalaxyID' in labels.columns:
    labels = labels.rename(columns={'GalaxyID':'img_id'})
sample_ids = labels['img_id'].astype(str).head(10).tolist()
print("Sample ids:", sample_ids)
for sid in sample_ids:
    p = TRAIN_IMG_DIR / f"{sid}.jpg"
    print(sid, "exists?", p.exists())


# In[36]:


from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (4,4)

WORK_DIR = Path(r"C:/Users/Sumana Sarkar/galaxy_data")
TRAIN_IMG_DIR = Path(r"C:/Users/Sumana Sarkar/galaxy_data/images_training/images_training_rev1")
LABELS_CSV = WORK_DIR / "labels" / "training_solutions_rev1.csv"

print("WORK_DIR:", WORK_DIR.exists())
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR, "exists?", TRAIN_IMG_DIR.exists(), "jpg_count:", len(list(TRAIN_IMG_DIR.glob("*.jpg"))))
print("Labels CSV exists?", LABELS_CSV.exists())


# In[37]:


labels = pd.read_csv(LABELS_CSV)
# unify id column
if 'GalaxyID' in labels.columns:
    labels = labels.rename(columns={'GalaxyID':'img_id'})
labels['img_id'] = labels['img_id'].astype(str)

# detect vote columns and top-answer
exclude = {'img_id','GalaxyID','Id','ID','id'}
vote_cols = [c for c in labels.columns if c not in exclude]
labels['top_answer_col'] = labels[vote_cols].idxmax(axis=1)

# map Class1.1/1.2/1.3 -> elliptical/spiral/other (conservative)
def map_top(col):
    s = str(col).lower()
    if 'class1.1' in s: return 'elliptical'
    if 'class1.2' in s: return 'spiral'
    if 'class1.3' in s: return 'other'
    if 'smooth' in s: return 'elliptical'
    if any(k in s for k in ['feature','disk','arm','spiral']): return 'spiral'
    return 'other'

labels['target'] = labels['top_answer_col'].apply(map_top)
print(labels['target'].value_counts())


# In[38]:


# default filenames: "<GalaxyID>.jpg"
labels['img_filename'] = labels['img_id'].astype(str)
# build paths and check existence
labels['img_path'] = labels['img_filename'].apply(lambda fn: TRAIN_IMG_DIR / f"{fn}.jpg")
labels['file_exists'] = labels['img_path'].apply(lambda p: p.exists())
print("Files found:", labels['file_exists'].sum(), "/", len(labels))

# show a couple of sample images (first available)
sample = labels[labels['file_exists']].head(6)
for idx, row in sample.iterrows():
    display(Image.open(row['img_path']))
    print(row['img_id'], row['target'])


# In[39]:


from sklearn.model_selection import train_test_split

SAMPLE_MODE = True   # -> True for fast iteration. Set False to use full dataset.
MAX_PER_CLASS = 2000  # per-class cap when SAMPLE_MODE True

df_available = labels[labels['file_exists']].copy()
if SAMPLE_MODE:
    df_use = df_available.groupby('target', group_keys=False).apply(lambda g: g.sample(n=min(len(g), MAX_PER_CLASS), random_state=42)).reset_index(drop=True)
else:
    df_use = df_available.reset_index(drop=True)

df_use = df_use[['img_filename','target']].rename(columns={'img_filename':'img_file'})
print("Using dataset rows:", len(df_use))
train_df, val_df = train_test_split(df_use, test_size=0.2, stratify=df_use['target'], random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print("train/val sizes:", train_df.shape, val_df.shape)


# In[40]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128,128)   # quick; change to (224,224) later for better performance
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    horizontal_flip=True,
    zoom_range=0.15,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

def make_gen(df, datagen, shuffle=True):
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=str(TRAIN_IMG_DIR),
        x_col='img_file',
        y_col='target',
        target_size=IMG_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        validate_filenames=True
    )

train_gen = make_gen(train_df, train_datagen, shuffle=True)
val_gen   = make_gen(val_df, val_datagen, shuffle=False)


# In[41]:


from pathlib import Path
TRAIN_IMG_DIR = Path(r"C:/Users/Sumana Sarkar/galaxy_data/images_training/images_training_rev1")

# print counts and a small filename sample
all_jpg = sorted([p.name for p in TRAIN_IMG_DIR.glob("*.jpg")])
print("Total JPG files in TRAIN_IMG_DIR:", len(all_jpg))
print("First 40 filenames (sample):")
for fn in all_jpg[:40]:
    print(" ", fn)


# In[42]:


from collections import defaultdict
from pathlib import Path
TRAIN_IMG_DIR = Path(r"C:/Users/Sumana Sarkar/galaxy_data/images_training/images_training_rev1")

# map exact stem -> filename (stem = basename without suffix)
stem_to_filenames = defaultdict(list)
for p in TRAIN_IMG_DIR.glob("*.jpg"):
    stem = p.stem  # e.g., '100008' or 'img_100008' or '00100008'
    stem_to_filenames[stem].append(p.name)

# quick stats
print("Unique stems found:", len(stem_to_filenames))
# show some stems sample
sample_stems = list(stem_to_filenames.keys())[:30]
print("Sample stems:", sample_stems)


# In[43]:


import pandas as pd
from pathlib import Path
TRAIN_IMG_DIR = Path(r"C:/Users/Sumana Sarkar/galaxy_data/images_training/images_training_rev1")
labels = pd.read_csv(Path(r"C:/Users/Sumana Sarkar/galaxy_data/labels/training_solutions_rev1.csv"))

# unify id column
if 'GalaxyID' in labels.columns:
    labels = labels.rename(columns={'GalaxyID':'img_id'})
labels['img_id'] = labels['img_id'].astype(str)

# helper to test candidate stems
def find_filename_for_id(imgid, stems_map, folder):
    candidates = []
    # exact
    if imgid in stems_map:
        candidates.extend(stems_map[imgid])
    # prefixed
    pref = f"img_{imgid}"
    if pref in stems_map:
        candidates.extend(stems_map[pref])
    # uppercase prefix
    pref2 = f"IMG_{imgid}"
    if pref2 in stems_map:
        candidates.extend(stems_map[pref2])
    # suffix _0
    suff = f"{imgid}_0"
    if suff in stems_map:
        candidates.extend(stems_map[suff])
    # zero-padded attempts
    for width in (7,8,9,10):
        try:
            padded = f"{int(imgid):0{width}d}"
            if padded in stems_map:
                candidates.extend(stems_map[padded])
        except:
            pass
    # if still empty, do a substring search: any filename that contains the id
    if not candidates:
        for stem, names in stems_map.items():
            if imgid in stem:
                candidates.extend(names)
    # return the first candidate (deterministic sort)
    if candidates:
        return sorted(set(candidates))[0]
    return None

# build stems_map
from collections import defaultdict
stems_map = defaultdict(list)
for p in TRAIN_IMG_DIR.glob("*.jpg"):
    stems_map[p.stem].append(p.name)

# apply to the first N rows (or entire df)
labels['matched_filename'] = labels['img_id'].apply(lambda i: find_filename_for_id(i, stems_map, TRAIN_IMG_DIR))

# summary stats
matched_count = labels['matched_filename'].notna().sum()
total = len(labels)
print(f"Matched filenames: {matched_count} / {total}  ({matched_count/total:.2%})")

# show first few matched & unmatched examples
print("\nMatched examples:")
print(labels[labels['matched_filename'].notna()][['img_id','matched_filename']].head(10))
print("\nUnmatched examples (first 10):")
print(labels[labels['matched_filename'].isna()][['img_id']].head(10))


# In[44]:


# attach coarse label 'target' (same mapping we used earlier)
vote_cols = [c for c in labels.columns if c not in ('img_id','matched_filename')]
labels['top_answer_col'] = labels[[c for c in labels.columns if c.startswith('Class')]].idxmax(axis=1)
def map_top(col):
    s = str(col).lower()
    if 'class1.1' in s: return 'elliptical'
    if 'class1.2' in s: return 'spiral'
    if 'class1.3' in s: return 'other'
    if 'smooth' in s: return 'elliptical'
    if any(k in s for k in ['feature','disk','arm','spiral']): return 'spiral'
    return 'other'
labels['target'] = labels['top_answer_col'].apply(map_top)

# keep only found files
df_found = labels[labels['matched_filename'].notna()].copy()
df_found['img_file'] = df_found['matched_filename']  # exact filename including extension
print("Rows with actual files:", len(df_found))

# show per-class counts
print("Per-class counts:")
print(df_found['target'].value_counts())

# if you want a fast sample for iteration:
SAMPLE_MODE = True
CAP_PER_CLASS = 2000
if SAMPLE_MODE:
    df_use = df_found.groupby('target', group_keys=False).apply(lambda g: g.sample(n=min(len(g), CAP_PER_CLASS), random_state=42)).reset_index(drop=True)
else:
    df_use = df_found.reset_index(drop=True)

print("Using rows for modeling:", len(df_use))
df_use = df_use[['img_file','target']].rename(columns={'img_file':'img_file'})


# In[45]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df_use, test_size=0.2, stratify=df_use['target'], random_state=42)
print("train/val sizes:", train_df.shape, val_df.shape)

IMG_SIZE = (128,128)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.12,
                                   height_shift_range=0.12, horizontal_flip=True, zoom_range=0.15, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=str(TRAIN_IMG_DIR),
    x_col='img_file',
    y_col='target',
    target_size=IMG_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    validate_filenames=True
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=str(TRAIN_IMG_DIR),
    x_col='img_file',
    y_col='target',
    target_size=IMG_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False,
    validate_filenames=True
)


# In[47]:


# Inspect the generator to get number of classes and mapping
print("Train generator samples:", getattr(train_gen, "samples", "unknown"))
print("Train generator batch_size:", getattr(train_gen, "batch_size", "unknown"))

# class_indices is the recommended way
print("class_indices:", train_gen.class_indices)
num_classes = len(train_gen.class_indices)
print("num_classes:", num_classes)

# Now build EfficientNetB0 model (same as before)
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0

IMG_SIZE = (128,128)   # keep this until you switch to final runs
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[48]:


import tensorflow as tf

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_galaxy_model.h5', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,   # short first; increase later
    callbacks=callbacks
)

model.save('galaxy_classifier_base.h5')


# In[50]:


import matplotlib.pyplot as plt

# history from model.fit -> variable `history`
plt.figure(figsize=(6,3))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss'); plt.show()

plt.figure(figsize=(6,3))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title('Accuracy'); plt.show()


# In[51]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
preds = model.predict(val_gen, steps=val_steps, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
target_names = list(val_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=target_names))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)


# In[52]:


# Inspect generators
print("train_gen.class_indices:", train_gen.class_indices)
import numpy as np
unique, counts = np.unique(train_gen.classes, return_counts=True)
print("train class counts (by index):", dict(zip(unique, counts)))
# map index->label
idx_to_label = {v:k for k,v in train_gen.class_indices.items()}
print("index->label:", idx_to_label)

unique_val, counts_val = np.unique(val_gen.classes, return_counts=True)
print("val class counts (by index):", dict(zip(unique_val, counts_val)))


# In[53]:


# examine one batch
x_batch, y_batch = next(train_gen)
print("x_batch.shape:", x_batch.shape)
print("y_batch.shape (one-hot):", y_batch.shape)
# show label distribution in the batch
labels_in_batch = y_batch.argmax(axis=1)
import numpy as np
print("batch label counts:", dict(zip(*np.unique(labels_in_batch, return_counts=True))))
# show mapping of columns -> label names
print("label index -> name:", {i:name for name,i in train_gen.class_indices.items()})


# In[54]:


from PIL import Image
# sample one filename per class from df_use (the dataframe you built earlier)
for label in train_df['target'].unique():
    file_sample = train_df[train_df['target']==label].iloc[0]['img_file']
    print(label, file_sample)
    display(Image.open(TRAIN_IMG_DIR / file_sample).resize((224,224)))


# In[56]:


from tensorflow.keras import layers, models, optimizers
import tensorflow as tf

# 1) Locate base (EfficientNetB0) used in your model. We assume 'base' variable exists.
# If not, find it like:
# for layer in model.layers:
#     print(layer.name, type(layer))

# Freeze base
base = None
for layer in model.layers:
    # find first layer that is not the input and has 4D output -> likely base
    if len(getattr(layer, "output_shape", ())) == 4:
        base = layer
        break

if base is None:
    # fallback if base variable exists already in your session
    base = model.layers[1]

base.trainable = False
print("Base layer found:", base.name)

# 2) Build a new head on top of the frozen base
inputs = model.input
x = base(inputs, training=False) if hasattr(base, '__call__') else model.layers[1].output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(train_gen.class_indices), activation='softmax')(x)

head_model = models.Model(inputs, outputs)

# 3) (Important) reinitialize the new Dense head weights (they are fresh by construction)
# Compile with a moderately higher LR for the head so it learns quickly
head_model.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# 4) Train the head for a bit (treat this as warm-up)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('head_only_best.h5', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

history_head = head_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,   # train the head a bit longer
    callbacks=callbacks
)

# Save the head-trained model
head_model.save('galaxy_head_trained.keras')
print("Head-only training finished.")


# In[58]:


# attach base from head_model construction
# If you used `base` above, it still points to the base layer.
base.trainable = True

# Freeze all but last N layers of the backbone
N = 40  # try 20 if GPU memory is tight; adjust
for i, layer in enumerate(base.layers):
    layer.trainable = True if i >= (len(base.layers) - N) else False

# Recompile at small LR
head_model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Fine-tune
fine_callbacks = [
    tf.keras.callbacks.ModelCheckpoint('finetuned_best.h5', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
]

fine_history = head_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=fine_callbacks
)
head_model.save('galaxy_finetuned_partial.keras')
print("Partial fine-tuning finished.")


# In[59]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
preds = model.predict(val_gen, steps=val_steps, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
target_names = list(val_gen.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=target_names))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)


# In[ ]:




