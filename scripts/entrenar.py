# ======================================================================
# Autor: Víctor Sierra Vicén
# Archivo: entrenar.py
# Descripción: Fine-tuning de GPT-2 español usando HuggingFace Trainer.
#              Compatible con GPU (Google Colab) y CPU.
# Uso: python scripts/entrenar.py
# ======================================================================

import json
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset


# ======================================================================
# CONFIGURACIÓN
# ======================================================================
MODELO_BASE = "DeepESP/gpt2-spanish"
DATOS_PATH = os.path.join("data", "procesado", "train.json")
TOKENIZER_PATH = os.path.join("data", "procesado", "tokenizer")
OUTPUT_DIR = "modelo"

# ======================================================================
# DATASET
# ======================================================================
class TransformerQADataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.datos = json.load(f)
        print(f"Dataset: {len(self.datos)} ejemplos", flush=True)

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }
    
# ======================================================================
# ENTRENAMIENTO
# ======================================================================
def entrenar():
    print("=" * 70, flush=True)
    print("ENTRENAMIENTO DEL MODELO", flush=True)
    print("=" * 70, flush=True)

    # Comprueba si hay GPU disponible, si no usa CPU
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"Dispositivo: {device}", flush=True)

    # Cargo el tokenizer guardado en preparar_datos.py y el modelo GPT-2
    print("Cargando tokenizer y modelo...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE,
        torch_dtype=torch.float32
    )

    #Le digo al modelo que haga espacio para los nuevos tokens
    modelo.resize_token_embeddings(len(tokenizer))

    num_params = sum(p.numel() for p in modelo.parameters())
    print(f"Parámetros: {num_params:,}", flush=True)

    # Creo una instancia de la clase TransformerQADataset
    dataset = TransformerQADataset(DATOS_PATH)

    # Data collator: crea labels automáticamente para modelos causales
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Son los argumentos que he ido cambiando a lo largo del entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=50, # Cuántas veces recorre el dataset completo                  
        per_device_train_batch_size=4, # Cuántos ejemplos procesa el modelo a la vez
        learning_rate=3e-5,  # Learning rate visto en el documento             
        warmup_steps=50,     # Pasos iniciales donde el learning rate aumenta gradualmente            
        weight_decay=0.05,   # Regularización para evitar el overfitting
        max_grad_norm=0.5,   # Gradient clipping, reescala la norma del gradiente
        logging_steps=10,   # Cada cuántos pasos se imprime la información del entrenamiento
        save_strategy="epoch",  # Guarda un checkpoint al final de cada época (epoch)
        save_total_limit=2, # Guarda solo los últimos dos checkpoints en disco
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42, # Fija la semilla para que el entrenamiento sea reproducible
    )

    # Uso la clase de Hugging Face que se encarga del entrenamiento 
    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=dataset,
    )

    # Entreno el modelo
    print("\nIniciando entrenamiento...", flush=True)
    print("-" * 70, flush=True)
    trainer.train()

    # Guardo modelo final ya entrenado
    print("\nGuardando modelo final...", flush=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Guardo el histórico de loss para poder generar la gráfica
    log_history = trainer.state.log_history
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    with open(os.path.join(OUTPUT_DIR, "historico_loss.json"), "w") as f:
        json.dump(losses, f)

    print(f"\nModelo guardado en: {OUTPUT_DIR}", flush=True)
    print(f"Loss final: {losses[-1]:.4f}", flush=True)
    print("Entrenamiento completado.", flush=True)


if __name__ == "__main__":
    entrenar()
