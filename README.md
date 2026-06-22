# Chatbot sobre el Modelo Transformer

Este proyecto es un chatbot basado en un modelo GPT-2, fine-tuneado para responder preguntas específicas sobre el modelo Transformer. Ha sido desarrollado como parte de mi Trabajo de Fin de Grado (TFG) de Matemáticas en la Universidad de Zaragoza.

## Autor
- **Víctor Sierra Vicén**

## Descripción del Proyecto
El objetivo de este chatbot es actuar como un asistente experto en el paper "Attention Is All You Need", presentado por Vaswani et al. en el 2017. El modelo GPT-2 ha sido entrenado con un dataset específico de preguntas y respuestas sobre la arquitectura, mecanismos y conceptos clave del modelo Transformer.

## Requisitos
- Python 3.10 o superior
- Las dependencias se encuentran en `requirements.txt`. Para instalarlas, ejecuta:
  ```bash
  pip install -r requirements.txt
  ```

## Uso del Proyecto

### 1. Preparar los Datos
Antes de entrenar, es necesario procesar el dataset. El script `preparar_datos.py` se encarga de tokenizar y formatear el dataset `dataset_completo.json` para el entrenamiento.

```bash
python scripts/preparar_datos.py
```
Los datos procesados se guardan en `data/procesado/`.

### 2. Entrenar el Modelo
Una vez los datos están listos, puedes entrenar el modelo. El script `entrenar.py` carga los datos procesados y fine-tunea el modelo GPT-2.
Se recomienda utilizar GPU para el entrenamiento del modelo debido a la larga duración de dicho proceso.

```bash
python scripts/entrenar.py
```
El modelo entrenado, junto con los checkpoints, se guardará en el directorio `modelo/` el cual es creado al ejecutar `entrenar.py`.

### 3. Chatear con el Modelo
Con el modelo ya entrenado, puedes interactuar con él a través de la consola.

```bash
python scripts/chatbot.py
```

### 4. Generar Gráfica de Pérdida (Opcional)
Si has guardado el historial de pérdida (`historico_loss.json`) durante el entrenamiento, puedes generar una gráfica para visualizar cómo ha evolucionado la función de pérdida durante el aprendizaje.

```bash
python scripts/generar_curva_loss.py
```

## Estructura General del Proyecto

```
TFG-chatbot/
│
├── data/
│   ├── dataset_completo.json         # Dataset original en formato JSON
│   └── procesado/
│       └── train.json                # Dataset procesado y listo para entrenar
│
├── scripts/
│   ├── preparar_datos.py             # Script para preprocesar el dataset
│   ├── entrenar.py                   # Script para fine-tunear el modelo GPT-2
│   ├── chatbot.py                    # Script para interactuar con el modelo entrenado
│   └── generar_curva_loss.py         # Script para visualizar la curva de loss
│
├── modelo/
│   ├── config.json                   # Configuración del modelo
│   ├── model.safetensors             # Pesos del modelo entrenado
│   └── ...                           # Otros archivos generados durante el entrenamiento
│
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Este archivo
```

### Directorios
- **`data/`**: Contiene el dataset de preguntas y respuestas.
  - `dataset_completo.json`: El dataset original.
  - `procesado/`: Contiene los datos una vez han sido procesados por `preparar_datos.py`.
- **`scripts/`**: Contiene todo el código fuente para procesar datos, entrenar y usar el chatbot.
- **`modelo/`**: Directorio donde se guarda el modelo fine-tuneado y los checkpoints generados durante el entrenamiento. Este directorio se crea al ejecutar `entrenar.py`.

## Licencia
Este proyecto no tiene licencia. Siéntete libre de usarlo como referencia para tus propios proyectos.
