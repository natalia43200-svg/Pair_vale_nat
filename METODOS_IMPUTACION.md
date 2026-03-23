# Métodos de Imputación de Valores Nulos

## Métodos Simples vs. KNN

### **Métodos Simples (Media, Mediana, Moda)**

#### Cuándo SÍ usarlos:
- **Pocos valores faltantes** (<5-10% del total)
- **Datos simples** sin relaciones complejas entre variables
- **Variables independientes** (ej: una columna aislada)
- **Variables categóricas**: usar **moda** (valor más frecuente)
- **Variables numéricas simples**: media (si distribución normal) o mediana (si sesgada)

#### Ejemplo:
```python
# Pocos datos faltantes, columna independiente
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
df['edad'] = imputer.fit_transform(df[['edad']])
```

#### Ventajas:
✅ Simple y rápido  
✅ Interpretable  
✅ Requiere menos recursos computacionales  

#### Desventajas:
❌ Ignora relaciones entre variables  
❌ Puede distorsionar distribuciones  
❌ No captura patrones complejos  

---

### **KNN Imputer (K-Nearest Neighbors)**

#### Por qué es MEJOR en datasets complejos:

1. **Múltiples dimensiones**: 27 columnas correlacionadas
   - Media/mediana ignora relaciones entre características
   - KNN captura que "países similares tienen valores similares"

2. **Datos heterogéneos**: PIB, esperanza de vida, densidad, etc.
   - Son muy diferentes entre sí
   - Un país pobre vs. rico tendrá valores distintos en múltiples columnas
   - KNN detecta patrones: "si este país es pobre, sus otras variables seguirán un patrón similar"

3. **Contexto geográfico/socioeconómico**:
   - Países de Similar desarrollo → valores similares
   - Media global distorsionaría los datos regionales

#### Ejemplo:
```python
from sklearn.impute import KNNImputer

columnas_num = ['density', 'gdp', 'life_expectancy', 'population', ...]
imputer_knn = KNNImputer(n_neighbors=5)
df[columnas_num] = imputer_knn.fit_transform(df[columnas_num])
```

#### Ventajas:
✅ Captura relaciones complejas entre variables  
✅ Usa similitud entre observaciones  
✅ Mantiene distribuciones y correlaciones  
✅ Mejor para datasets multidimensionales  

#### Desventajas:
❌ Más lento que métodos simples  
❌ Requiere más recursos computacionales  
❌ Puede "suavizar" valores extremos  

---

### **IterativeImputer (Imputación Iterativa)**

#### Cuándo usarlo:
- Relaciones no-lineales muy complejas
- Patrones iterativos necesarios para mejorar predicciones
- Datasets con muchas variables interdependientes

#### Ejemplo:
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10)
df[columnas_num] = imputer.fit_transform(df[columnas_num])
```

---

## Tabla Comparativa

| Método | Usa | NO usa | Ejemplo |
|--------|-----|--------|---------|
| **Media** | Datos normales, pocos nulos | Outliers, datos sesgados | Sueldos sin sesgo |
| **Mediana** | Datos sesgados, robusto | Relaciones complejas | Datos con outliers |
| **Moda** | Variables **categóricas** | Variables numéricas | País, género, región |
| **KNN** | Múltiples dimensiones correlacionadas | Datasets muy simples | **27 variables socioeconómicas** |
| **IterativeImputer** | Patrones iterativos muy complejos | Datasets pequeños | Relaciones no-lineales |

---

## Guía de Decisión

### Pregúntate:

1. **¿Cuántos valores faltantes hay?**
   - <5% → Métodos simples pueden funcionar
   - >10% → Necesitas métodos más sofisticados

2. **¿Las columnas están relacionadas?**
   - Sí → KNN o IterativeImputer
   - No (independientes) → Media/Mediana

3. **¿Qué tipo de datos tengo?**
   - Numéricos → Media/Mediana/KNN/IterativeImputer
   - Categóricos → Moda
   - Mixtos → KNN (maneja ambos)

4. **¿Cuál es el contexto del dataset?**
   - Datos socioeconómicos, geográficos, biomédicos → KNN
   - Datos simples, independientes → Media/Mediana

---

## En este Proyecto

**Dataset**: Datos mundiales (world_data_full.csv) con 27 columnas numéricas correlacionadas

**Decisión**: **KNN Imputer** ✅
- 27 variables altamente correlacionadas
- Contexto socioeconómico donde similitud importa
- Más de 10% de datos faltantes en algunas columnas
- Preserva patrones y relaciones reales

