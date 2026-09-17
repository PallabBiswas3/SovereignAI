TSDIAG CARE v6 KAGGLE BUNDLE
================================

Purpose
-------
Run the canonical CARE-to-Compare v6 Wind SCADA benchmark on Kaggle. The runner
accepts either the original ZIP or the directory layout Kaggle creates when it
expands an uploaded archive. The normal-behavior model is fitted on each event's
healthy training split and evaluated on its prediction split. This is benchmark
fitting/evaluation, not supervised deep-model training.

Upload
------
1. Upload tsdiag-care-kaggle.zip as one private Kaggle Dataset.
2. Create a second private Kaggle Dataset from the official CARE Zenodo remote
   URL, or upload a complete CARE_To_Compare.zip. Do not use the incomplete
   local download. Kaggle normally expands uploaded archives automatically.
3. Create a Kaggle Notebook and attach both datasets.
4. Import kaggle/care_benchmark.ipynb from this bundle, or copy its cells into
   the notebook.
5. Keep the notebook private if CARE's distribution terms require it.

Run sequence
------------
The notebook starts with a three-event smoke test. When that succeeds, change
RUN_MODE from "smoke" to "full". Enable VERIFY_MD5 only if the original ZIP is
visible under /kaggle/input; Kaggle-expanded layouts cannot reproduce the ZIP's
byte-for-byte checksum.

Command-line equivalent inside Kaggle:

  python kaggle/run_care_kaggle.py --mode smoke
  python kaggle/run_care_kaggle.py --mode full --verify-md5

Outputs
-------
/kaggle/working/care_results/wind_scada_benchmark.json
/kaggle/working/care_results/wind_scada_event_table.csv
/kaggle/working/care_results/care_v6_summary.md
/kaggle/working/care_results/kaggle_run_manifest.json

Download all four artifacts. A valid publishable run has zero failures, one
provenance row per successful event, and canonical CUSUM values 0.5 / 10.0 / 6.
When the original ZIP remains visible, also require the official CARE MD5
2547b58c21ac8c242d13232860cf500c. Otherwise retain the private Kaggle Dataset
URL/version and the runner's three-farm layout validation as provenance.
