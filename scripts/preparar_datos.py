# ======================================================================
# Autor: Víctor Sierra Vicén
# Archivo: preparar_datos.py
# Descripción: Carga el dataset JSON, formatea cada par 
#              pregunta-respuesta como un prompt para GPT-2 y guarda el 
#              resultado tokenizado.
# Uso: python scripts/preparar_datos.py
# ======================================================================

import json
import os
from transformers import AutoTokenizer

# ======================================================================
# CONFIGURACIÓN
# ======================================================================
MODELO_BASE = "DeepESP/gpt2-spanish"
DATASET_PATH = os.path.join("data", "dataset_completo.json")
OUTPUT_DIR = os.path.join("data", "procesado")
MAX_LENGTH = 128  # Longitud máxima en tokens (por ejemplo, se puede cambiar)

# Tokens especiales para delimitar pregunta y respuesta
TOKEN_PREGUNTA = "<pregunta>"
TOKEN_RESPUESTA = "<respuesta>"
TOKEN_FIN = "<fin>" #Para indicar fin de secuencia


# Carga el dataset json
def cargar_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        datos = json.load(f)
    print(f"Dataset cargado: {len(datos)} pares pregunta-respuesta")
    return datos


# Convierte una pareja pregunta-respuesta en el formato que el modelo
# aprende a generar.
# Formato: <pregunta> ... <respuesta> ... <fin>
def formatear_prompt(pregunta, respuesta):
    return f"{TOKEN_PREGUNTA} {pregunta} {TOKEN_RESPUESTA} {respuesta} {TOKEN_FIN}"


# Carga un tokenizer de español estándar y le añade los tokens especiales
def preparar_tokenizer(modelo_base):
    tokenizer = AutoTokenizer.from_pretrained(modelo_base)
    
    # GPT-2 no tiene pad_token por defecto, usamos eos_token
    # Esto es para si una frase es más corta que la longitud máxima,
    # se rellena con un token especial de padding 
    tokenizer.pad_token = tokenizer.eos_token
    
    # Añado los tokens especiales creados
    tokens_nuevos = [TOKEN_PREGUNTA, TOKEN_RESPUESTA, TOKEN_FIN]
    num_nuevos = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens_nuevos} #Lo paso como diccionario
    )
    print(f"Tokens especiales añadidos: {num_nuevos}")
    print(f"Vocabulario total: {len(tokenizer)} tokens")
    
    return tokenizer

# Formatea y tokeniza todos los elementos del database
# Marca qué tokens pertenecen a la respuesta para que el modelo
# solo aprenda a generar esa parte
def tokenizar_dataset(datos, tokenizer, max_length):
    ejemplos_formateados = []
    ejemplos_descartados = 0
    
    # ID del token que marca el inicio de la respuesta
    token_resp_id = tokenizer.convert_tokens_to_ids(TOKEN_RESPUESTA)
    
    for item in datos: # Un item es un par pregunta-respuesta
        prompt = formatear_prompt(item["pregunta"], item["respuesta"])
        
        encoding = tokenizer( #Tokenizo el prompt
            prompt,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze().tolist()
        attention_mask = encoding["attention_mask"].squeeze().tolist()
        
        # Verifico longitud, si se ha truncado lo descartamos porque está incompleto
        tokens_sin_padding = sum(attention_mask)
        if tokens_sin_padding >= max_length:
            ejemplos_descartados += 1
            continue
        
        # Creo labels: -100 en la pregunta y el padding,
        # solo los tokens de la respuesta contribuyen a la función de pérdida
        labels = [-100] * len(input_ids)
        
        # Encontramos dónde empieza la respuesta
        try:
            idx_resp = input_ids.index(token_resp_id)
            # Los tokens desde <respuesta> hasta el final 
            # (excluyendo padding) son los que el modelo aprende
            for i in range(idx_resp + 1, len(input_ids)):
                if attention_mask[i] == 1: #Si no es de padding 
                    labels[i] = input_ids[i]
        except ValueError:
            ejemplos_descartados += 1
            continue
        
        ejemplos_formateados.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })
    
    print(f"Ejemplos procesados: {len(ejemplos_formateados)}")
    if ejemplos_descartados > 0:
        print(f"Ejemplos descartados: {ejemplos_descartados}")
    
    return ejemplos_formateados

# Guarda los datos tokenizados y el tokenizer modificado
def guardar_datos(ejemplos, tokenizer, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardo los datos tokenizados
    output_path = os.path.join(output_dir, "train.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ejemplos, f)
    print(f"Datos guardados en: {output_path}")
    
    # Guardo el tokenizer con los tokens especiales
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer guardado en: {tokenizer_path}")

# Muestra un ejemplo formateado para verificar
def mostrar_ejemplo(datos, tokenizer):  
    item = datos[0]
    prompt = formatear_prompt(item["pregunta"], item["respuesta"])
    tokens = tokenizer.encode(prompt)
    
    print("\n" + "=" * 70)
    print("EJEMPLO DE VERIFICACIÓN")
    print("=" * 70)
    print(f"\nPregunta: {item['pregunta']}")
    print(f"Respuesta: {item['respuesta']}")
    print(f"\nPrompt formateado:\n{prompt}")
    print(f"\nNúmero de tokens: {len(tokens)}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens[:20])}...")
    print("=" * 70)


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PREPARANDO EL DATASET...")
    print("=" * 70)
    
    datos = cargar_dataset(DATASET_PATH)
    tokenizer = preparar_tokenizer(MODELO_BASE)
    mostrar_ejemplo(datos, tokenizer)
    ejemplos = tokenizar_dataset(datos, tokenizer, MAX_LENGTH)
    guardar_datos(ejemplos, tokenizer, OUTPUT_DIR)
    print("\nPREPARACIÓN COMPLETADA")
