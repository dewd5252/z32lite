# Colab Extension Quickstart (VS Code)

هذا المسار لتشغيل كل شيء من VS Code باستخدام إضافة Colab.

## المتطلبات
- امتداد `google.colab` مثبت.
- امتداد `ms-toolsai.jupyter` مثبت.
- فتح هذا المشروع كـ workspace في VS Code.

## خطوات سريعة جدًا
1. افتح أي ملف Notebook (`.ipynb`) داخل المشروع.
2. أعلى اليمين اختر `Select Kernel`.
3. اختر `Colab` ثم `Auto Connect`.
4. سجّل دخول Google عند ظهور طلب المصادقة.
5. من Command Palette شغّل: `Colab: Open Terminal` (لو لم يظهر فعّل `colab.terminal` من إعدادات workspace).
6. داخل Terminal الخاص بـ Colab نفّذ:

```bash
cd /content
git clone <YOUR_REPO_URL> z32lite || true
cd z32lite
./finetune/colab_oneclick.sh balanced /content/z32lite_runs
```

## مخرجات متوقعة
- `dataset/processed/train.json`, `holdout.json`, `eval.json`
- `dataset/processed/qwen_jsonl/*.jsonl`
- Model artifacts تحت:
  - `/content/z32lite_runs/balanced/final_merged`
  - `/content/z32lite_runs/gguf/z32lite_f16.gguf`
  - `/content/z32lite_runs/gguf/z32lite_Q4_K_M.gguf`

## تقييم النموذج
بعد توليد predictions على `eval.json`:

```bash
python3 dataset/score_outputs.py --predictions predictions.jsonl --report eval_report.json
```
