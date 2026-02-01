# Proyecto: Implementación y evaluación de modelos de aprendizaje supervisado en Python
# Universidad de Especializades Espiritu Santo
# Maestria en Inteligencia Artificial

Repositorio para la materia de **Aprendizaje Automatico** - Maestria en Inteligencia Artificial, UEES.

---
Estudiante:
Ingeniero Gonzalo Mejia Alcivar

Ingeniero Jorge Ortiz Merchan

Docente: Ingeniera GLADYS MARIA VILLEGAS RUGEL

Fecha de Ultima Actualizacion: 01 Febrero 2026

---

## Analisis del Dominio del DataSet

### Objetivo

Desarrollar un modelo de Machine Learning de clasificacion que permita categorizar a las empresas del Ecuador segun su nivel de desempeno financiero (**alto**, **medio** y **bajo**), utilizando datos financieros historicos, sectoriales y geograficos, con el fin de apoyar la toma de decisiones estrategicas en los ambitos financiero, empresarial y de gestion economica.

### Descripcion del Dominio

El dataset proviene de registros financieros de empresas del Ecuador correspondientes al ano 2024, con un total de **134,865 registros**. Los datos abarcan dos sectores regulatorios principales:

| Sector | Registros |
|---|---|
| SOCIETARIO | 134,458 |
| MERCADO DE VALORES | 407 |

El dominio se enmarca en el analisis financiero empresarial ecuatoriano, donde la Superintendencia de Companias, Valores y Seguros recopila informacion contable y financiera de las empresas bajo su supervision.

### Variables del DataSet

El dataset contiene **11 variables** que describen la estructura financiera y operativa de cada empresa:

| Variable | Descripcion | Tipo |
|---|---|---|
| Ano | Periodo fiscal del reporte | Categorica |
| Sector | Sector regulatorio (Societario / Mercado de Valores) | Categorica |
| Cant. Empleados | Numero de empleados de la empresa | Numerica |
| Activo | Total de activos de la empresa | Numerica |
| Patrimonio | Patrimonio neto de la empresa | Numerica |
| IngresoVentas | Ingresos generados por ventas | Numerica |
| UtilidadAntesImpuestos | Utilidad bruta antes de impuestos | Numerica |
| UtilidadEjercicio | Utilidad del ejercicio fiscal | Numerica |
| UtilidadNeta | Utilidad neta despues de deducciones | Numerica |
| IR_Causado | Impuesto a la Renta causado | Numerica |
| IngresosTotales | Total de ingresos de la empresa | Numerica |

### Contexto del Problema

La clasificacion de empresas por desempeno financiero es un problema relevante en el ambito economico del Ecuador por las siguientes razones:

- **Toma de decisiones estrategicas:** Permite a inversionistas, reguladores y gestores identificar rapidamente el estado financiero de una empresa.
- **Politicas publicas:** Facilita a entidades gubernamentales el diseno de politicas de apoyo o fiscalizacion diferenciada segun el nivel de desempeno.
- **Gestion de riesgo:** Ayuda a instituciones financieras a evaluar el riesgo crediticio de las empresas.
- **Benchmarking sectorial:** Posibilita la comparacion entre empresas del mismo sector para identificar mejores practicas.

### Enfoque de Clasificacion

La variable objetivo (target) sera construida a partir de indicadores financieros derivados del dataset, categorizando a las empresas en tres niveles de desempeno:

- **Alto:** Empresas con indicadores financieros superiores (alta rentabilidad, buena estructura patrimonial).
- **Medio:** Empresas con indicadores financieros dentro del rango promedio del sector.
- **Bajo:** Empresas con indicadores financieros por debajo del promedio o con resultados negativos.

### Tecnicas de Machine Learning Aplicables

Al tratarse de un problema de **clasificacion multiclase supervisada**, se evaluaran modelos como:

- Arboles de Decision
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines (SVM)
- Regresion Logistica Multinomial
- Redes Neuronales (MLP)

La seleccion del modelo final dependera de metricas de evaluacion como accuracy, precision, recall, F1-score y la matriz de confusion.

---

## Analisis Exploratorio de Datos (EDA)

> Script: [`scr/1_ExploracionEDA.py`](scr/1_ExploracionEDA.py)

### Resumen del Dataset

- **Registros analizados:** 134,865
- **Variables originales:** 11 (10 numericas + 1 categorica)
- **Valores nulos:** 0
- **Sectores:** SOCIETARIO (134,458) | MERCADO DE VALORES (407)

### Variable Objetivo Creada: Desempeno Financiero

Se creo la variable **Desempeno** a partir del **Margen Neto** (UtilidadNeta / IngresosTotales), clasificando a las empresas en tres niveles mediante cuantiles:

| Nivel | Cantidad | Porcentaje |
|---|---|---|
| Bajo | 72,006 | 53.39% |
| Alto | 44,955 | 33.33% |
| Medio | 17,904 | 13.28% |

Adicionalmente se derivaron los indicadores **ROA** (Rentabilidad sobre Activos) y **ROE** (Rentabilidad sobre Patrimonio).

### Hallazgos Principales

- Las variables financieras presentan **alta asimetria positiva** (pocas empresas grandes concentran valores elevados), por lo que se aplico escala logaritmica en las visualizaciones.
- El **coeficiente de variacion** supera 14x en la mayoria de variables, reflejando la gran heterogeneidad del tejido empresarial ecuatoriano.
- Existe **alta correlacion** entre Activo, Patrimonio, IngresoVentas e IngresosTotales, lo que sugiere la necesidad de seleccion de features o reduccion de dimensionalidad.
- Los indicadores derivados (Margen Neto, ROA, ROE) muestran **separacion clara entre clases**, validando su utilidad como predictores.

### Visualizaciones Generadas

#### 1. Distribucion de la Variable Objetivo

![Distribucion Variable Objetivo](results/01_distribucion_variable_objetivo.png)

#### 2. Distribucion por Sector

![Distribucion por Sector](results/02_distribucion_sector.png)

#### 3. Histogramas de Variables Financieras

![Histogramas Variables Financieras](results/03_histogramas_variables_financieras.png)

#### 4. Boxplots de Indicadores por Desempeno

![Boxplots Indicadores Desempeno](results/04_boxplots_indicadores_desempeno.png)

#### 5. Boxplots de Variables Financieras por Desempeno

![Boxplots Variables Financieras](results/05_boxplots_variables_financieras.png)

#### 6. Matriz de Correlacion

![Matriz de Correlacion](results/06_matriz_correlacion.png)

#### 7. Pairplot de Indicadores Clave

![Pairplot Indicadores](results/07_pairplot_indicadores.png)

#### 8. Estadisticas Descriptivas por Clase

![Tabla Estadisticas por Clase](results/08_tabla_estadisticas_por_clase.png)

---

## Preprocesamiento de Datos

> Script: [`scr/2_PreProcesamiento.py`](scr/2_PreProcesamiento.py)

### Pasos Realizados

#### 1. Eliminacion de columna Año

Se elimino la columna **Año** del dataset ya que contiene un unico valor (2024) y no aporta poder predictivo al modelo. El dataset paso de 11 a **10 columnas**.

#### 2. Tratamiento de valores nulos

Se realizo un diagnostico completo de valores nulos en el dataset:

- **Nulos encontrados:** 0
- **Estrategia definida:** Mediana para variables numericas, moda para categoricas (aplicable si se detectaran nulos tras conversion de tipos)

#### 3. Codificacion de variables categoricas

Se aplico **LabelEncoder** a la variable categorica **Sector**:

| Valor Original | Codigo |
|---|---|
| MERCADO DE VALORES | 0 |
| SOCIETARIO | 1 |

Variable objetivo **Desempeno**:

| Nivel | Codigo |
|---|---|
| Alto | 0 |
| Bajo | 1 |
| Medio | 2 |

#### 4. Escalado de variables numericas

Se aplico **StandardScaler** (estandarizacion Z-score) a las 10 features del modelo, transformando cada variable para tener **media = 0** y **desviacion estandar = 1**.

Features escaladas:
`Sector`, `Cant_Empleados`, `Activo`, `Patrimonio`, `IngresoVentas`, `UtilidadAntesImpuestos`, `UtilidadEjercicio`, `UtilidadNeta`, `IR_Causado`, `IngresosTotales`

#### 5. Division en conjunto de entrenamiento y prueba (80/20)

Se realizo una division **estratificada** para mantener la proporcion de clases en ambos conjuntos:

| Conjunto | Registros | Porcentaje |
|---|---|---|
| Entrenamiento | 107,892 | 80% |
| Prueba | 26,973 | 20% |

Distribucion de clases (verificacion de estratificacion):

| Clase | Entrenamiento | Prueba |
|---|---|---|
| Bajo | 53.39% | 53.39% |
| Alto | 33.33% | 33.33% |
| Medio | 13.28% | 13.28% |

### Visualizaciones del Preprocesamiento

#### 9. Comparativa Antes/Despues del Escalado

![Comparativa Escalado](results/09_comparativa_escalado.png)

#### 10. Distribucion Train/Test por Clase

![Distribucion Train Test](results/10_distribucion_train_test.png)

#### 11. Distribucion de Features Escaladas

![Distribucion Features Escaladas](results/11_distribucion_features_escaladas.png)

#### 12. Resumen del Preprocesamiento

![Resumen Preprocesamiento](results/12_resumen_preprocesamiento.png)
