# Aprendizaje de la dinámica de un péndulo

Este proyecto compara dos redes que aproximan la dinámica de un péndulo simple ideal. La MLP aprende directamente el campo vectorial; la HNN aprende un Hamiltoniano y obtiene el campo mediante sus derivadas. Ambas usan los mismos datos, particiones y protocolo.

## Presentación interactiva

Abre [Aprender el péndulo (MLP vs HNN).html](<PAGÍNA WEB/Aprender el péndulo (MLP vs HNN).html>) en un navegador. Conserva [mlp-hnn-weights.js](<PAGÍNA WEB/mlp-hnn-weights.js>) en la misma carpeta: contiene los pesos MLP-64 y HNN-64 usados por la simulación. MathJax se carga desde internet.

## Física y modelos

El estado es \(x=(q,p)\), donde \(q\) es el ángulo y \(p\) el momento angular. Para \(m=l=1\) y \(g=9.81\):

\[
H(q,p)=\frac{p^2}{2ml^2}+mgl(1-\cos q),\qquad
\dot q=\frac{\partial H}{\partial p},\quad
\dot p=-\frac{\partial H}{\partial q}.
\]

La MLP produce \((\dot q,\dot p)\). La HNN produce un escalar \(H_\theta\) y usa \((\partial H_\theta/\partial p,-\partial H_\theta/\partial q)\). El valor absoluto de \(H_\theta\) queda indeterminado hasta una constante; lo físicamente relevante para el campo son sus derivadas.

## Datos y resultados auditados

`data/` conserva únicamente 200 condiciones iniciales de entrenamiento y 200 de prueba. Todas pertenecen al régimen oscilatorio. Los CSV planos de 40 200 estados se regeneran con RK4, paso de 0.02 s y horizonte de 4 s; se excluyen de Git porque ocupan más de 10 MiB sin aportar información nueva.

Los resultados compactos están en [results-summary.csv](results-summary.csv) y [results-by-domain.csv](results-by-domain.csv). Las métricas de \(\dot q\) y \(\dot p\) se informan por separado porque tienen unidades incompatibles y no deben promediarse. En la ejecución incluida:

- MLP-128 obtuvo el menor RMSE de \(\dot q\).
- HNN-128 obtuvo el menor RMSE de \(\dot p\) y la menor deriva media de energía física.
- MLP-64 obtuvo los menores RMSE de rollout para \(q\) y \(p\).

Son resultados descriptivos de una sola semilla y de un único conjunto de prueba; no estiman variabilidad entre inicializaciones ni cubren el régimen rotacional. Los pesos y cifras de la presentación proceden de la ejecución auditada anterior. Esa ejecución usó una tolerancia absoluta que no siempre conservó el mínimo de validación observado; el código actual ya separa la selección del checkpoint del criterio de paciencia. Por ello, no deben interpretarse como el óptimo alcanzable de cada arquitectura y una reproducción nueva puede mejorar las cifras.

En el análisis ID/OOD, OOD sólo significa que la energía inicial queda fuera del intervalo observado en las 180 trayectorias de ajuste, \([0.1033,12.0962]\) J. Las 200 trayectorias de prueba provienen del mismo procedimiento de muestreo; no es una prueba general de cambio de distribución.

## Instalación y uso

Se requiere Python 3.10 o superior. Desde la raíz:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Genera los CSV planos antes de entrenar:

```powershell
python -m pendulum_project generate --data-dir data --output-dir data
```

Entrena y evalúa ambas redes con 64 unidades por capa:

```powershell
python -m pendulum_project all --n-train 200 --models mlp hnn --hidden-dim 64 --device cpu --output-dir results-reproduccion
```

Para repetir el barrido de anchuras:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
foreach ($width in @(32, 64, 128)) {
    python -m pendulum_project all --n-train 200 --models mlp hnn `
        --hidden-dim $width --device cpu `
        --output-dir "results-reproduccion\width-$width"
}
```

Los comandos `train`, `evaluate` y `all` aceptan `--models mlp`, `--models hnn` o ambos. `sweep --sizes 100 150 200` compara tamaños de entrenamiento. Los datos planos, checkpoints, figuras e historiales son regenerables y quedan ignorados por Git.

## Estructura mínima

```text
data/                       Condiciones iniciales compactas
src/pendulum_project/       Física, datos, modelos, entrenamiento y evaluación
tests/                      Pruebas de física, datos, modelos y métricas
PAGÍNA WEB/                 Presentación y pesos necesarios para la simulación
results-*.csv               Resumen numérico auditado
```
