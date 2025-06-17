
# ✅ Módulo 1: Imports y configuración
import os, torch
import pandas as pd
import numpy as np
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from transformers import set_seed

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Dispositivo: {device}")

# ✅ Módulo 2: Carga de datos y preparación de clases
df = pd.read_excel("Emociones_super_revisadas_test_II.xlsx")
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["emotion"])
label2id = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
id2label = {v: k for k, v in label2id.items()}

# Pesos por clase para manejar desbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df["label"]), y=df["label"])
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"\n📊 Pesos de clase:")
for k, v in zip(label_encoder.classes_, class_weights):
    print(f"{k}: {v:.2f}")

# ✅ Módulo 3: Split estratificado
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
train_dataset = Dataset.from_pandas(df_train[["text", "label"]])
val_dataset = Dataset.from_pandas(df_val[["text", "label"]])

# ✅ Módulo 4: Tokenización y entrenamiento para múltiples modelos
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, EvalPrediction, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

modelos = [
    "roberta-base",
    "microsoft/deberta-v3-base",
    "google/electra-base-discriminator",
    "xlnet-base-cased"
]

for nombre_modelo in modelos:
    nombre_salida = nombre_modelo.split("/")[-1].replace("-", "_")
    output_dir = f"./resultados/{nombre_salida}_optuna"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    config = AutoConfig.from_pretrained(nombre_modelo, num_labels=len(label2id), id2label={int(k): v for k, v in id2label.items()}, label2id={k: int(v) for k, v in label2id.items()})

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(nombre_modelo, config=config)

    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        report = classification_report(labels, preds, target_names=[id2label[i] for i in sorted(id2label.keys())], output_dict=True)
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision_macro": precision_score(labels, preds, average="macro"),
            "recall_macro": recall_score(labels, preds, average="macro"),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "precision_weighted": precision_score(labels, preds, average="weighted"),
            "recall_weighted": recall_score(labels, preds, average="weighted"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }
        for emocion, i in label2id.items():
            if emocion in report:
                metrics[f"eval_f1_{emocion.lower()}"] = report[emocion]["f1-score"]
        return metrics

    def build_args(best_hp=None):
        return TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_strategy="epoch",
            num_train_epochs=best_hp["num_train_epochs"] if best_hp else 10,
            per_device_train_batch_size=best_hp["per_device_train_batch_size"] if best_hp else 8,
            per_device_eval_batch_size=8,
            learning_rate=best_hp["learning_rate"] if best_hp else 2e-5,
            weight_decay=best_hp["weight_decay"] if best_hp else 0.1,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            report_to="none",
            logging_dir=f"{output_dir}/logs",
            seed=SEED
        )

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 10),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
        }

    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    args_temp = build_args()

    trainer_temp = WeightedLossTrainer(
        model_init=model_init,
        args=args_temp,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    best_run = trainer_temp.hyperparameter_search(
        direction="maximize",
        hp_space=optuna_hp_space,
        n_trials=5,
        compute_objective=lambda metrics: metrics["eval_f1_macro"],
        backend="optuna"
    )

    args = build_args(best_run.hyperparameters)

    trainer = WeightedLossTrainer(
        model_init=model_init,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    trainer.train()

    with open(f"{output_dir}/optuna_mejor_config.txt", "w") as f:
        f.write("🧪 Mejores hiperparámetros encontrados por Optuna:\n\n")
        for key, value in best_run.hyperparameters.items():
            f.write(f"{key}: {value}\n")

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Modelo entrenado y guardado en: {output_dir}")
