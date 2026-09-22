# Travail realise (toi + moi)

Ce dossier `Guilhem` contient uniquement ce que nous avons construit pour le challenge MIASHS 2026.

## Contenu

- `sdm/`: package PyTorch multi-modal (data, modele, loss, train, inference, pseudo-labeling).
- `scripts/`:
  - `audit_modalities.py`: audit de couverture des modalites (patch / series / tabulaire) par split.
  - `train_pa.py`: entrainement Teacher sur PA avec spatial block CV.
  - `pseudo_label_po.py`: pseudo-labellisation PO (positif / negatif / ignore).
  - `train_student.py`: entrainement Student sur PA + PO pseudo-labellise.
  - `calibrate_set_size.py`: calibration OOF de `alpha` et `max_k`.
  - `predict_test_ensemble.py`: inference test avec moyenne des folds.
  - `predict_test.py`: inference test et generation `submission.csv`.
  - `long_run_script.py`: orchestrateur autonome (resume/retry/logs/lock).
  - `validate_submission.py`: validation du format Kaggle.
- `configs/default.yaml`: configuration centralisee (chemins, training, pseudo-labels, inference).
- `tests/test_smoke_pipeline.py`: smoke test bout-en-bout sur donnees synthetiques.
- `requirements.txt`: dependances (avec ajout de `PyYAML`).
- `README.md`: documentation globale mise a jour avec commandes de run.

## Execution rapide (serveur rtx4)

```bash
python3 scripts/long_run_script.py \
  --data-root ./data \
  --config configs/default.yaml \
  --output-root ./runs/v31 \
  --folds 0 1 2 3 4 \
  --gpu-id 0 \
  --max-retries 1 \
  --resume \
  --stop-on-error
```

## Remarques

- Le pipeline est concu pour 1 GPU + 4 CPU.
- `long_run_script.py` applique les bornes CPU (threads BLAS = 1), gere lock anti-doublon, resume et logs JSON.
- V3.2: `train_student.py` applique un reequilibrage PO tabulaire-only via `sampling.po_tabular_only_factor`, limite PO via `training.max_po_samples`, et effectue un refine final PA-only (`student_refine`).
- La validation interne est en spatial block CV.
- La sortie de soumission respecte le format Kaggle (`surveyId,predictions`, especes triees, non-vide).
