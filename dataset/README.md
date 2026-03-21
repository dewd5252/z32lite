# Z32LITE Dataset Pipeline

الـ dataset بقى مبني كـ pipeline واضح بدل ملف واحد للتجارب.

## المخرجات
- `dataset/processed/train.json`
- `dataset/processed/holdout.json`
- `dataset/processed/eval.json`
- `dataset/processed/manifest.json`
- `dataset/z32lite_dataset.json`

## schema
كل مثال يحتوي على:
- `id`
- `split`
- `category`
- `language`
- `source`
- `tags`
- `conversations`

وفي `eval.json` يوجد أيضًا `evaluation` لاستخدامه في التقييم الآلي.

## الفئات
- `conversational_core`
- `tool_calling`
- `refusal_safety`
- `boundary_knowledge`

## أوامر العمل
بناء البيانات:

```bash
python3 dataset/build_dataset.py
```

التحقق:

```bash
python3 dataset/validate_dataset.py
```

تصدير JSONL بصيغة Qwen لاستخدامها في Colab:

```bash
python3 dataset/export_qwen_jsonl.py
```

تقييم مخرجات موديل على `eval.json`:

```bash
python3 dataset/score_outputs.py --predictions path/to/predictions.jsonl
```

## ملاحظات
- التدريب نفسه مفترض أن يتم من Colab أو VS Code extension المتصلة بـ Colab.
- `z32lite_dataset.json` موجود فقط للتوافق مع الـ workflow القديم.
- `train.json` و`holdout.json` و`eval.json` هي الملفات المرجعية الجديدة.
