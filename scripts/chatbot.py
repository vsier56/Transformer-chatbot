# ======================================================================
# Autor: Víctor Sierra Vicén
# Archivo: chatbot.py
# Descripción: Interfaz de terminal para conversar con el modelo 
#              entrenado.
# Uso: python scripts/generar_curva_loss.py
# ======================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================================================================
# CONFIGURACIÓN
# ======================================================================
MODELO_DIR = "modelo"

# Parámetros de generación
MAX_NEW_TOKENS = 150       # Máximo de tokens a generar
TEMPERATURE = 0.3          # Creatividad (0 = determinista, 1 = aleatorio)
TOP_K = 30                 # Considera los top-k tokens más probables
TOP_P = 0.85               # Nucleus sampling
REPETITION_PENALTY = 1.5   # Penalizar repeticiones


# Tokens especiales (deben coincidir con preparar_datos.py)
TOKEN_PREGUNTA = "<pregunta>"
TOKEN_RESPUESTA = "<respuesta>"
TOKEN_FIN = "<fin>"

# Carga el modelo entrenado y el tokenizer
def cargar_modelo():
    print("Cargando modelo...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODELO_DIR)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_DIR)
    
    # GPT-2 no tiene pad_token por defecto
    tokenizer.pad_token = tokenizer.eos_token
    
    modelo.eval()  # Modo inferencia
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    
    print(f"Modelo cargado en {device}.")
    return modelo, tokenizer, device

# Genera una respuesta a una pregunta. Formatea la entrada como en el 
# entrenamiento y deja que el modelo complete la respuesta.
def generar_respuesta(pregunta, modelo, tokenizer, device):

    # Construye el prompt únicamente de la pregunta
    prompt = f"{TOKEN_PREGUNTA} {pregunta} {TOKEN_RESPUESTA}"
    
    # Tokeniza la pregunta
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Genera la respuesta
    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids(TOKEN_FIN)
        )
    
    # Decodifica solo los tokens generados (sin el prompt)
    tokens_generados = outputs[0][inputs["input_ids"].shape[1]:]
    respuesta = tokenizer.decode(tokens_generados, skip_special_tokens=False)
    
    # Quita el token <fin> y texto posterior a este
    if TOKEN_FIN in respuesta:
        respuesta = respuesta[:respuesta.index(TOKEN_FIN)]
    
    return respuesta.strip()

# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 70)
    print("CHATBOT SOBRE EL MODELO TRANSFORMER")
    print("TFG Matemáticas - Víctor Sierra Vicén")
    print(70)
    print()
    
    modelo, tokenizer, device = cargar_modelo()
    
    print("\nEscribe tu pregunta sobre el Transformer.")
    print("Escribe 'salir' para terminar.\n")
    print("-" * 70)
    
    while True:
        pregunta = input("\nTú: ").strip()
        
        if pregunta.lower() in ["salir", "exit", "quit", "Salir"]:
            print("\n¡Hasta luego!")
            break
        
        if not pregunta:
            continue
        
        respuesta = generar_respuesta(pregunta, modelo, tokenizer, device)
        print(f"\nBot: {respuesta}")
        print("-" * 70)


if __name__ == "__main__":
    main()
