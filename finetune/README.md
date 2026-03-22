# Z32LITE Fine-tuning Workflow

الـ workflow الجديد **Colab-first**. المقصود إن تجهيز البيانات يتم من الريبو، لكن التدريب والتصدير النهائي يتمان من Google Colab أو VS Code extension المتصلة بـ Colab.

## 1. جهّز البيانات

```bash
python3 dataset/build_dataset.py
python3 dataset/validate_dataset.py
python3 dataset/export_qwen_jsonl.py
```

## 2. ارفع الملفات إلى Colab
الملفات الأساسية:
- `dataset/processed/qwen_jsonl/train.jsonl`
- `dataset/processed/qwen_jsonl/holdout.jsonl`
- `dataset/processed/eval.json`
- `finetune/train_qlora.py`
- `finetune/export_gguf.py`
- `finetune/requirements-colab.lock.txt`

## 3. ثبّت المتطلبات داخل Colab

```bash
pip install -r finetune/requirements-colab.lock.txt
```

## 4. شغّل التدريب

```bash
python finetune/train_qlora.py \
  --train-file dataset/processed/qwen_jsonl/train.jsonl \
  --eval-file dataset/processed/qwen_jsonl/holdout.jsonl \
  --profile balanced \
  --precision fp16
```

اختبار توافق سريع بدون تدريب فعلي:

```bash
python finetune/train_qlora.py \
  --train-file dataset/processed/qwen_jsonl/train.jsonl \
  --eval-file dataset/processed/qwen_jsonl/holdout.jsonl \
  --profile balanced \
  --precision fp16 \
  --dry-run
```

بديل أسرع (أمر واحد يشغّل كامل الـ pipeline):

```bash
./finetune/colab_oneclick.sh balanced /content/z32lite_runs
```

بديل بدون كتابة أوامر يدوية:
- افتح `finetune/z32lite_colab_oneclick.ipynb`
- شغّل `Runtime > Run all`

## 5. صدّر إلى GGUF

```bash
python finetune/export_gguf.py \
  --model-dir /content/z32lite_runs/balanced/final_merged \
  --output-dir /content/z32lite_runs/gguf
```

## 6. قيّم النتائج
بعد inference على `dataset/processed/eval.json` وتوليد ملف predictions:

```bash
python dataset/score_outputs.py --predictions predictions.jsonl
```

## profiles المتاحة
- `balanced`
- `tool_heavy`
- `light_regularization`
- `colab_safe` (fallback أخف تلقائيًا في حال فشل profile أساسي)

## ملاحظات
- لا يوجد local training path في هذه الخطة.
- `colab_oneclick.sh` يشغّل preflight قبل التدريب ويكتب `preflight.json` و`pipeline_status.json`.
- سياسة التدريب الرسمية على Colab T4: `fp16`.
- لو شغال عبر VS Code extension على Colab، استخدم نفس الأوامر داخل terminal الجلسة.
- احتفظ بنسخة `fp16` merged قبل quantization للمقارنة.
- تشغيل VS Code extension خطوة بخطوة موجود في:
  - `finetune/COLAB_EXTENSION_QUICKSTART.md`
