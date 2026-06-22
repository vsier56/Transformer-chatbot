# ======================================================================
# Autor: Víctor Sierra Vicén
# Archivo: generar_curva_loss.py
# Descripción: Genera la gráfica de la evolución de la función de 
#              pérdida durante el entrenamiento del chatbot.
# Uso: python scripts/generar_curva_loss.py
# ======================================================================

import json
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

# Cargo el archivo de modelo/historico_loss.json
LOSS_PATH = os.path.join("modelo", "historico_loss.json")

with open(LOSS_PATH, "r") as f:
    losses = json.load(f)

# Generar eje x (steps o epochs según el formato)
# En nuestro caso, serán steps
n = len(losses)

# Detecta si son steps o epochs
if n > 100:
    # Son steps, calcular epochs aproximados
    xlabel = "Step de entrenamiento"
    x = list(range(1, n + 1))
else:
    # Son epochs
    xlabel = "Época"
    x = list(range(1, n + 1))

# Gráfica
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, losses, color='#2563eb', linewidth=1.5)
ax.set_xlabel(xlabel, fontsize=14)
ax.set_ylabel('Función de pérdida ($\\mathcal{L}$)', fontsize=14)
ax.set_title('Evolución de la pérdida durante el entrenamiento', 
             fontsize=14, fontweight='bold')

# Señalo pérdida inicial y final
ax.annotate(f'Loss inicial: {losses[0]:.3f}', 
            xy=(x[0], losses[0]),
            xytext=(x[0] + len(x)*0.1, losses[0]),
            fontsize=13, color='#dc2626',
            arrowprops=dict(arrowstyle='->', color='#dc2626'))

ax.annotate(f'Loss final: {losses[-1]:.4f}', 
            xy=(x[-1], losses[-1]),
            xytext=(x[-1] - len(x)*0.4, losses[-1] + (max(losses) - min(losses))*0.15),
            fontsize=13, color='#16a34a',
            arrowprops=dict(arrowstyle='->', color='#16a34a'))

ax.grid(True, alpha=0.3)

plt.tight_layout()
# Guardo en ambos formatos
plt.savefig('loss_curve.pdf', format='pdf', dpi=300) 
plt.savefig('loss_curve.png', format='png', dpi=300)
plt.close()

print(f"Puntos registrados: {n}")
print(f"Loss inicial: {losses[0]:.4f}")
print(f"Loss final: {losses[-1]:.4f}")
print(f"Loss mínimo: {min(losses):.4f}")
print("Gráficas guardadas: loss_curve.pdf, loss_curve.png")
